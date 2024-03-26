import asyncio
from typing import List, Dict, Set, Callable, Tuple

import networkx as nx

from algos.diffusion import leak_diffusion
from daos.contract import ContractDao, ContractCompileItem
from daos.money import MoneyTransferDao, TransferItem
from daos.trace import PCTraceItem, PCTraceDao
from downloaders.contract import ContractSourceDownloader, ContractBytecodeDownloader
from downloaders.trace import PCTraceDownloader
from downloaders.trans import TransactionDownloader
from utils.bucket import AsyncItemBucket
from utils.typed_ast import get_ast_graph


class FL4SCGraph:
    def __init__(
            self, rpc_bucket: AsyncItemBucket, apikey_bucket: AsyncItemBucket,
            gamma: float = 0.1, epsilon: float = 1e-3,
            grad_func: Callable = lambda x: x ** 3,
            target_element_type: str = 'FunctionDefinition',
    ):
        self.rpc_bucket = rpc_bucket
        self.apikey_bucket = apikey_bucket
        self.gamma = gamma
        self.epsilon = epsilon
        self.grad_func = grad_func
        self.target_element_type = target_element_type

    async def process(self, dapp: Dict) -> nx.DiGraph:
        # load all pc
        txhashs = list()
        for _txhashs in dapp['fault']['transaction_hash']:
            txhashs.extend(_txhashs)
        txhashs.extend(dapp['faultless']['transaction_hash'])
        txhash2pc_list = await self._load_pctrace_data(txhashs)

        # load all compilation result (fault only)
        contract_addresses = set()
        for _txhashs in dapp['fault']['transaction_hash']:
            for _txhash in txhashs:
                pc_list = txhash2pc_list[_txhash]
                contract_addresses.update([pc_item.address for pc_item in pc_list])
        addr2compilation = await self._load_compile_result(contract_addresses)

        # load all ast related to the case
        related_addr_and_fn = set()
        addr_pc2source_map = dict()
        for addr, compilation in addr2compilation.items():
            for item in compilation.source_mapping:
                addr_pc = '{}#{}'.format(addr, item.pc)
                addr_pc2source_map[addr_pc] = item
        for pc_list in txhash2pc_list.values():
            for pc_item in pc_list:
                addr_pc = '{}#{}'.format(pc_item.address, pc_item.pc)
                mapping_item = addr_pc2source_map.get(addr_pc)
                if mapping_item is None:
                    continue
                related_addr_and_fn.add((pc_item.address, mapping_item.filename))
        graph = await self._load_ast_graph(related_addr_and_fn, addr2compilation)

        # init target nodes
        target_nodes = [
            node for node, attr in graph.nodes(data=True)
            if attr.get('type') == self.target_element_type
        ]

        # source mapping and money tracing
        txhash_pkgs = list()
        txhash_pkgs.extend(dapp['fault']['transaction_hash'])
        txhash_pkgs.extend([[_txhash] for _txhash in dapp['faultless']['transaction_hash']])
        pkg_idx2fault = [
            True if i < len(dapp['fault']['transaction_hash']) else False
            for i in range(len(txhash_pkgs))
        ]
        for i, _txhashs in enumerate(txhash_pkgs):
            vis_pc = set()
            case_node = 'case_{}'.format(i)
            graph.add_node(case_node, type='Case', is_fault=pkg_idx2fault[i])
            pc_list = list()
            for _txhash in _txhashs:
                pc_list.extend(txhash2pc_list[_txhash])

            # link the case node to ast nodes
            for pc_item in reversed(pc_list):
                addr_pc = '{}#{}'.format(pc_item.address, pc_item.pc)
                if addr_pc in vis_pc:
                    continue
                vis_pc.add(addr_pc)

                mapping_item = addr_pc2source_map.get(addr_pc)
                if mapping_item is None:
                    continue
                ast_node = '{}#{}#{}:{}'.format(
                    pc_item.address, mapping_item.filename,
                    mapping_item.begin, mapping_item.offset,
                )
                if not graph.nodes.get(ast_node):
                    continue
                graph.add_edge(
                    case_node, ast_node,
                    type='cover',
                    pc_index=pc_item.index,
                )

            # add money features for the ast nodes
            if not pkg_idx2fault[i]:
                continue
            transfer_graph = await self._load_money_graph(
                transaction_hash_list=_txhashs,
                pc_list=pc_list,
            )

            # set transfer involved attr
            transfer_involved_addrs = set()
            for _, attr in transfer_graph.nodes(data=True):
                transfer: TransferItem = attr['info']
                transfer_involved_addrs.add(transfer.from_address)
                transfer_involved_addrs.add(transfer.to_address)
            for n in graph.nodes():
                addr = n.split('#')[0]
                if addr not in transfer_involved_addrs:
                    continue
                graph.nodes[n]['transfer_involved'] = True

            # set money leakage score
            transfer_idx2score = await self._leak_diffusion(transfer_graph)
            pc_item_and_score = [
                (transfer_graph.nodes[idx]['info'].pc_item, score)
                for idx, score in transfer_idx2score.items()
            ]
            for pc_item, score in pc_item_and_score:
                addr_pc = '{}#{}'.format(pc_item.address, pc_item.pc)
                mapping_item = addr_pc2source_map.get(addr_pc)
                if mapping_item is None:
                    continue

                addr_fn = '{}#{}'.format(pc_item.address, mapping_item.filename)
                for n in target_nodes:
                    if not n.startswith(addr_fn):
                        continue
                    _, _, src = n.split('#')
                    begin, offset = src.split(':')
                    begin, end = int(begin), int(begin) + int(offset)
                    if mapping_item.begin >= begin and \
                            mapping_item.begin + mapping_item.offset <= end:
                        graph.nodes[n]['leakage'] = graph.nodes[n].get('leakage', 0) + score
                        break

        # map fault location to contract graph
        for location in dapp['fault'].get('location', list()):
            address, filename, src = location.split('#')
            fault_begin, fault_offset = [int(offset) for offset in src.split(':')]
            fault_end = fault_begin + fault_offset
            addr_fn = '{}#{}'.format(address, filename)
            for n in target_nodes:
                if not n.startswith(addr_fn):
                    continue
                _, _, src = n.split('#')
                begin, offset = src.split(':')
                begin, end = int(begin), int(begin) + int(offset)
                if fault_begin >= begin and fault_end <= end:
                    graph.nodes[n]['is_fault'] = True
        return graph

    async def _load_pctrace_data(self, transaction_hashs: List[str]) -> Dict[str, List[PCTraceItem]]:
        async def _create_task(_transaction_hash: str):
            return await PCTraceDao(downloader=PCTraceDownloader(
                rpc_url=await self.rpc_bucket.get(),
            )).get_pc_list(transaction_hash=_transaction_hash)

        tasks = [_create_task(txhash) for txhash in transaction_hashs]
        result = await asyncio.gather(*tasks)
        txhash2pc_list = dict()
        for i, txhash in enumerate(transaction_hashs):
            txhash2pc_list[txhash] = result[i]
        return txhash2pc_list

    async def _load_compile_result(self, contract_addresses: Set[str]) -> Dict[str, ContractCompileItem]:
        async def _create_task(_address: str):
            return await ContractDao(downloader=ContractSourceDownloader(
                apikey=await self.apikey_bucket.get(),
            )).get_compile_item(_address)

        # load compile result of all contract
        tasks = [_create_task(addr) for addr in contract_addresses]
        result = await asyncio.gather(*tasks)
        return {addr: result[i] for i, addr in enumerate(contract_addresses)}

    async def _load_ast_graph(
            self, related_addr_and_fn: List[Tuple[str, str]],
            addr2compilation: Dict[str, ContractCompileItem]
    ) -> nx.DiGraph:
        graph = nx.DiGraph()

        async def _create_task(_address: str, _fn: str):
            ast = addr2compilation[_address].ast
            if not ast.get(_fn):
                return
            g = await get_ast_graph(ast[_fn])
            nodes = dict()
            for n, attr in g.nodes(data=True):
                begin, offset, _ = attr['src'].split(':')
                src = '{}:{}'.format(begin, offset)
                node_name = '{}#{}#{}'.format(_address, _fn, src)
                attr['filename'] = _fn
                nodes[n] = (node_name, attr)
            graph.add_nodes_from(nodes.values())
            graph.add_edges_from([
                (nodes[u][0], nodes[v][0], dict(type='child', order=attr['order']))
                for u, v, attr in g.edges(data=True)
            ])

        # load all result
        tasks = [_create_task(addr, fn) for addr, fn in related_addr_and_fn]
        await asyncio.gather(*tasks)
        return graph

    async def _load_money_graph(
            self, transaction_hash_list: List[str],
            pc_list: List[PCTraceItem],
    ) -> nx.DiGraph:
        return await MoneyTransferDao(downloader=TransactionDownloader(
            rpc_url=await self.rpc_bucket.get(),
        )).get_transfer_graph(
            transaction_hash_list=transaction_hash_list,
            pc_list=pc_list,
        )

    async def _leak_diffusion(self, transfer_graph: nx.DiGraph) -> Dict[int, float]:
        # async def _create_task(_address: str) -> bool:
        #     return await ContractDao(downloader=ContractBytecodeDownloader(
        #         rpc_url=await self.rpc_bucket.get()
        #     )).is_contract(contract_address=_address)

        # find the profit addresses from the zero out-degree nodes
        # profit_addr2transfer = {}
        # for node, attr in transfer_graph.nodes(data=True):
        #     addr = attr['info'].to_address
        #     if profit_addr2transfer.get(addr) is None:
        #         profit_addr2transfer[addr] = set()
        #     profit_addr2transfer[addr].add(node)

        # check whether the address is contract or not
        # tasks = [_create_task(addr) for addr in profit_addr2transfer.keys()]
        # result = await asyncio.gather(*tasks)
        # sources = set()
        # for i, addr in enumerate(profit_addr2transfer.keys()):
        #     if result[i]:
        #         continue
        #     sources = sources.union(profit_addr2transfer[addr])

        # 每个地址的入账 & 每个地址的利润
        addr2token2in = dict()
        addr2token2profit = dict()
        for node, attr in transfer_graph.nodes(data=True):
            item: TransferItem = attr['info']

            token2in = addr2token2in.get(item.to_address, dict())
            in_transfers = token2in.get(item.symbol, list())
            in_transfers.append(item)
            token2in[item.symbol] = in_transfers
            addr2token2in[item.to_address] = token2in

            token2profit = addr2token2profit.get(item.to_address, dict())
            token2profit[item.symbol] = token2profit.get(item.symbol, 0) + item.amount
            addr2token2profit[item.to_address] = token2profit

            token2profit = addr2token2profit.get(item.from_address, dict())
            token2profit[item.symbol] = token2profit.get(item.symbol, 0) - item.amount
            addr2token2profit[item.from_address] = token2profit

        # filter
        for addr, token2in in addr2token2in.items():
            for token, in_transfers in token2in.items():
                # 去掉所有利润/入账 < 5%的入账
                in_amount = sum([t.amount for t in in_transfers])
                if addr2token2profit[addr][token] / in_amount < 0.05:
                    addr2token2in[addr][token] = list()
                    continue

                # 去掉所有代币兑换的入账
                unswapped_transfers = list()
                for transfer in in_transfers:
                    edges = transfer_graph.out_edges(transfer.index)
                    if len(edges) != 0:
                        continue
                    unswapped_transfers.append(transfer)
                addr2token2in[addr][token] = unswapped_transfers

        # collect
        sources = set()
        for addr, token2in in addr2token2in.items():
            for token, in_transfers in token2in.items():
                for transfer in in_transfers:
                    sources.add(transfer.index)

        # execute leak diffusion
        return leak_diffusion(
            graph=transfer_graph,
            sources=sources,
            gamma=self.gamma,
            epsilon=self.epsilon,
            grad_func=self.grad_func,
        )
