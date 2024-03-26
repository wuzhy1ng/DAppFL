import asyncio
from typing import List

import networkx as nx

from daos.trace import PCTraceItem
from downloaders.defs import Downloader
from utils.web3 import hex_to_dec, parse_token_transfer


class TransferItem:
    def __init__(
            self, from_address: str, to_address: str,
            amount: int, index: int, symbol: str,
            pc_item: PCTraceItem,
    ):
        self.from_address = from_address
        self.to_address = to_address
        self.amount = amount
        self.index = index
        self.symbol = symbol
        self.pc_item = pc_item

    def __str__(self):
        return '{}->{} {} {} {}'.format(
            self.from_address, self.to_address,
            self.amount, self.symbol, self.index,
        )


class MoneyTransferDao:
    def __init__(self, downloader: Downloader):
        self.downloader = downloader

    async def _get_money_transfers(
            self, transaction_hash_list: List[str],
            pc_list: List[PCTraceItem]
    ) -> List[TransferItem]:
        transfers = list()
        trace, logs = list(), list()
        for transaction_hash in transaction_hash_list:
            result = await self.downloader.download(transaction_hash=transaction_hash)
            trace.extend(result['trace'])
            logs.extend(result['logs'])

            # the first trace item must be in the graph,
            # because it is an external transaction
            transfers.append(TransferItem(
                from_address=trace[0]['from'],
                to_address=trace[0]['to'],
                amount=hex_to_dec(trace[0].get('value')),
                index=0,
                symbol='0x' + '0' * 40,
                pc_item=PCTraceItem(
                    transaction_hash=transaction_hash, index=-1,
                    pc=-1, opcode='', depth=-1, address='', is_error=False,
                ),
            ))

        # extract transfer from trace and logs
        trace_idx, logs_idx = 1, 0
        is_trace_transfer = {
            op: True for op in [
                'CALL', 'CALLCODE', 'STATICCALL', 'DELEGATECALL',
                'CREATE', 'SELFDESTRUCT',
            ]
        }
        is_log = {'LOG%s' % i: True for i in range(4 + 1)}
        for i, pc_item in enumerate(pc_list):
            if is_trace_transfer.get(pc_item.opcode):
                transfers.append(TransferItem(
                    from_address=trace[trace_idx]['from'],
                    to_address=trace[trace_idx]['to'],
                    amount=hex_to_dec(trace[trace_idx].get('value')),
                    index=i,
                    symbol='0x' + '0' * 40,
                    pc_item=pc_item,
                ))
                trace_idx += 1
                continue

            if not is_log.get(pc_item.opcode) or logs_idx >= len(logs):
                continue
            token_transfer = parse_token_transfer(logs[logs_idx])
            logs_idx += 1
            if token_transfer is not None:
                transfers.append(TransferItem(
                    from_address=token_transfer['from'],
                    to_address=token_transfer['to'],
                    amount=hex_to_dec(token_transfer['value']),
                    index=i,
                    symbol=token_transfer['symbol'],
                    pc_item=pc_item,
                ))

        return transfers

    def _process_money_transfers(self, transfers: List[TransferItem]) -> List[TransferItem]:
        # reflect the token transfer pc item to caller
        zero_address = '0x' + '0' * 40
        naive_transfers = list()
        for i, transfer in enumerate(transfers):
            if transfer.symbol == zero_address:
                naive_transfers.append(transfer)
                continue
            if transfer.from_address == transfer.pc_item.address:
                continue
            for naive_transfer in reversed(naive_transfers):
                if naive_transfer.pc_item.address == transfer.from_address:
                    transfers[i].pc_item = naive_transfer.pc_item
                    break

        # filter zero amount transfer
        rlt = [transfer for transfer in transfers if transfer.amount > 0]
        for i in range(len(rlt)):
            rlt[i].index = i
        return rlt

    async def get_transfer_graph(
            self, transaction_hash_list: List[str],
            pc_list: List[PCTraceItem]
    ) -> nx.DiGraph:
        """
        Get a graph from the external transaction,
        internal transactions, and token transfers.

        :param transaction_hash_list: A list of transaction hash.
        :param pc_list: the pc list while transaction execution.
        :return: A graph.
        """
        # load transfer data
        transfers = await self._get_money_transfers(
            transaction_hash_list=transaction_hash_list,
            pc_list=pc_list
        )
        transfers = self._process_money_transfers(transfers)

        # extract all transfer by account
        addresses = set()
        addr2transfer_out = dict()
        addr2transfer_in = dict()
        for transfer in transfers:
            addresses.add(transfer.from_address)
            addresses.add(transfer.to_address)
            if not addr2transfer_out.get(transfer.from_address):
                addr2transfer_out[transfer.from_address] = list()
            addr2transfer_out[transfer.from_address].append(transfer)
            if not addr2transfer_in.get(transfer.to_address):
                addr2transfer_in[transfer.to_address] = list()
            addr2transfer_in[transfer.to_address].append(transfer)

        # build graph nodes
        g = nx.DiGraph()
        for transfer in transfers:
            g.add_node(transfer.index, info=transfer)

        # build graph edges by strategies
        for transfer in reversed(transfers):
            # token redirection
            if transfer.index - 1 >= 0 and transfers[transfer.index - 1].symbol != transfer.symbol:
                swap_transfer_idx = transfer.index - 1
                if transfer.from_address == transfers[swap_transfer_idx].to_address:
                    g.add_edge(
                        transfers[swap_transfer_idx].index,
                        transfer.index,
                        weight=1.0,
                    )
                elif transfer.to_address == transfers[swap_transfer_idx].from_address:
                    g.add_edge(
                        transfer.index,
                        transfers[swap_transfer_idx].index,
                        weight=1.0,
                    )
                    continue

            # weight pollution and temporal reasoning
            txs_in_linked = [
                tx_in for tx_in in addr2transfer_in.get(transfer.from_address, list())
                if tx_in.symbol == transfer.symbol and tx_in.index < transfer.index
            ]
            sum_link = sum([_tx.amount for _tx in txs_in_linked])
            for tx_in in txs_in_linked:
                g.add_edge(
                    tx_in.index, transfer.index,
                    weight=tx_in.amount / sum_link,
                )

        return g


async def test():
    # TODO: for more checking
    from downloaders.trans import TransactionDownloader
    from downloaders.trace import PCTraceDownloader
    from daos.trace import PCTraceDao
    d = PCTraceDao(downloader=PCTraceDownloader(
        'https://eth-mainnet.nodereal.io/v1/317f6d43dd4c4acea1fa00515cf02f90'
    ))
    pc_list = await d.get_pc_list('0xa992b28ecf2eed778d20d5200946ea341b950be0c3d78b1f2237a4d8d795de95')
    # pc_list.extend(await d.get_pc_list('0xcd314668aaa9bbfebaf1a0bd2b6553d01dd58899c508d4729fa7311dc5d33ad7'))
    d = MoneyTransferDao(downloader=TransactionDownloader(
        'https://eth-mainnet.nodereal.io/v1/317f6d43dd4c4acea1fa00515cf02f90'
    ))
    g = await d.get_transfer_graph(
        transaction_hash_list=[
            "0xa992b28ecf2eed778d20d5200946ea341b950be0c3d78b1f2237a4d8d795de95",
            # "0xb5c8bd9430b6cc87a0e2fe110ece6bf527fa4f170a4bc8cd032f768fc5219838"
        ],
        pc_list=pc_list
    )

    from matplotlib import pyplot as plt
    nx.draw(g, with_labels=True, pos=nx.shell_layout(g))
    plt.show()

    from algos.diffusion import leak_diffusion
    x = leak_diffusion(
        g, {7}, epsilon=1e-4,
        grad_func=lambda _x: _x * 5,
    )
    print(sorted(x.items(), key=lambda _x: _x[1], reverse=True))
    for i, attr in g.nodes(data=True):
        print(i, attr['info'], attr['info'].pc_item)


if __name__ == '__main__':
    asyncio.run(test())
