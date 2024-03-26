import asyncio
import datetime
import json
import os
from typing import Union, List, Tuple, Dict, Callable

import networkx as nx
import torch
from torch_geometric.data import Dataset, HeteroData

from dataset.nx import FL4SCGraph
from utils.bucket import AsyncItemBucket


class FL4SCDataset(Dataset):
    def __init__(
            self, root: str,
            net2rpc: Dict[str, List[str]], net2apikey: Dict[str, List[str]],
            leakage: bool = True, gamma: float = 0.1, epsilon: float = 1e-3,
            grad_func: Callable = lambda x: x ** 3,
            mapping: bool = True,
            target_element_type: str = 'FunctionDefinition',
            transform: Callable = None,
    ):
        self.leakage = leakage == bool('True')
        self.mapping = mapping == bool('True')
        self.gamma = gamma
        self.epsilon = epsilon
        self.grad_func = grad_func
        self._net2rpc_bucket = {
            net: AsyncItemBucket(items=_rpc_urls, qps=5)
            for net, _rpc_urls in net2rpc.items()
        }
        self._net2apikey_bucket = {
            net: AsyncItemBucket(items=_apikeys, qps=1)
            for net, _apikeys in net2apikey.items()
        }
        self.target_element_type = target_element_type
        self._data = dict()
        super().__init__(root, transform=transform)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        path = os.path.join(self.root, 'raw')
        return [os.path.join(path, fn) for fn in os.listdir(path)]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        path = os.path.join(self.root, 'raw')
        return [
            os.path.join(self.root, 'processed', '%s.pt' % fn.split('.')[0])
            for fn in os.listdir(path)
        ]

    @property
    def metadata(self) -> Tuple:
        node_types, edge_types = set(), set()
        if not getattr(self, '_metadata', None):
            for fn in self.processed_file_names:
                data = torch.load(os.path.join(self.processed_dir, fn))
                node_types.update(data.metadata()[0])
                edge_types.update(data.metadata()[1])
            self._metadata = (list(node_types), list(edge_types))
        return self._metadata

    def len(self) -> int:
        return len(self.raw_file_names)

    def get(self, idx: int) -> HeteroData:
        data = torch.load(self.processed_file_names[idx])
        if not self.leakage:
            for t in filter(lambda _t: _t != 'Case', data.node_types):
                data[t].x = data[t].x[:, :-2]
        if not self.mapping:
            data = data.node_type_subgraph([
                t for t in data.node_types
                if t != 'Case'
            ])
        return data

    def process(self):
        # load cases from raw files
        dapps = list()
        for fn in self.raw_file_names:
            with open(fn, 'r') as f:
                dapps.append(json.load(f))

        # build dataset
        for i, dapp in enumerate(dapps):
            print('{}: processing `{}` ({}/{})'.format(
                datetime.datetime.now(),
                dapp.get("name"),
                i + 1, len(dapps)
            ))
            if os.path.exists(self.processed_file_names[i]):
                continue
            async_func = lambda _dapp: FL4SCGraph(
                rpc_bucket=self._net2rpc_bucket[dapp['platform']],
                apikey_bucket=self._net2apikey_bucket[dapp['platform']],
                gamma=self.gamma, epsilon=self.epsilon,
                grad_func=self.grad_func,
            ).process(_dapp)
            data = asyncio.get_event_loop().run_until_complete(async_func(dapp))
            data = self.save_graph2data(data)
            torch.save(data, self.processed_file_names[i])
        print('{}: finished!'.format(datetime.datetime.now()))

    @staticmethod
    def save_graph2data(graph: nx.DiGraph) -> HeteroData:
        # load node features
        # two types of nodes, i.e., `Case` and `Code`,
        # where the case node has attribute `is_fault`,
        # and ast code nodes have `filename`, `src`, `attr`, and `is_fault`.
        data = HeteroData()
        node_name2idx = dict()
        node_type2features = dict()
        node_type2node_name = dict()

        # load case feats
        case_nodes = sorted([
            node for node, attr in graph.nodes(data=True)
            if attr.get('type') == 'Case'
        ])
        case_num = len(case_nodes)
        node_type2features['Case'] = list()
        fault_coverage_map = dict()
        non_fault_coverage_map = dict()
        for i, case_node in enumerate(case_nodes):
            attr = graph.nodes[case_node]
            node_name2idx[case_node] = i
            node_type2features['Case'].append([
                graph.degree(case_node),
                1 if attr.get('is_fault') else 0,
            ])
            for neighbor in graph.neighbors(case_node):
                if attr.get('is_fault'):
                    fault_coverage_map[neighbor] = fault_coverage_map.get(neighbor, 0) + 1
                    continue
                non_fault_coverage_map[neighbor] = non_fault_coverage_map.get(neighbor, 0) + 1

        # load other node feats
        # addrs = set([
        #     node_name.split('#')[0]
        #     for node_name in graph.nodes()
        #     if node_name.startswith('0x')
        # ])
        # addr2idx = {addr: idx for idx, addr in enumerate(list(addrs))}
        for node_name, attr in graph.nodes(data=True):
            node_type = attr.get('type')
            if node_type == 'Case':
                continue
            if node_type2features.get(node_type) is None:
                node_type2features[node_type] = list()
                node_type2node_name[node_type] = list()
            node_name2idx[node_name] = len(node_type2features[node_type])
            node_type2node_name[node_type].append(node_name)

            # start building the node features
            # (init with leakage score, degree, and coverage info)
            ast_attr = attr.get('attr', dict())
            node_type2features[node_type].append([
                graph.degree(node_name),

                # coverage features
                fault_coverage_map.get(node_name, 0),
                non_fault_coverage_map.get(node_name, 0),
                case_num - fault_coverage_map.get(node_name, 0),
                case_num - non_fault_coverage_map.get(node_name, 0),

                # contract features
                1 if ast_attr.get('fullyImplemented') else 0,

                # function features
                1 if ast_attr.get('isConstructor') else 0,
                1 if ast_attr.get('virtual') else 0,

                # statements features
                1 if ast_attr.get('stateVariable') else 0,
                1 if ast_attr.get('constant') else 0,

                # code location
                # addr2idx.get(node_name.split('#')[0], -1),
                # *[int(num) for num in attr['src'].split(':')],

                # money leakage features
                1 if attr.get('transfer_involved') else 0,
                attr.get('leakage', 0),
            ])

        # load node labels
        node_type2labels = dict()
        for node_name, attr in graph.nodes(data=True):
            node_type = attr.get('type')
            if node_type2labels.get(node_type) is None:
                node_type2labels[node_type] = list()
            node_type2labels[node_type].append(1 if attr.get('is_fault', False) else 0)

        # load edge features
        # edge contains `Case-AST` and other ast edges,
        # where `Case-AST` has `pc_index` attribute
        edge_type2edge_index = dict()
        edge_type2edge_attr = dict()
        max_pc_index = max([attr.get('pc_index', 0) for _, _, attr in graph.edges(data=True)])
        for u, v, attr in graph.edges(data=True):
            u2v_edge_type = (
                str(graph.nodes[u].get('type', '')),
                attr.get('type'),
                str(graph.nodes[v].get('type', '')),
            )
            if edge_type2edge_index.get(u2v_edge_type) is None:
                edge_type2edge_index[u2v_edge_type] = list()
            edge_type2edge_index[u2v_edge_type].append([
                node_name2idx[u], node_name2idx[v],
            ])
            if edge_type2edge_attr.get(u2v_edge_type) is None:
                edge_type2edge_attr[u2v_edge_type] = list()
            edge_attr = [attr.get('pc_index', 0) / max_pc_index] \
                if attr['type'] == 'cover' else [attr.get('order')]
            edge_type2edge_attr[u2v_edge_type].append(edge_attr)

            # add rev edge
            v2u_edge_type = (
                str(graph.nodes[v].get('type', '')),
                '_%s' % attr.get('type'),
                str(graph.nodes[u].get('type', '')),
            )
            if edge_type2edge_index.get(v2u_edge_type) is None:
                edge_type2edge_index[v2u_edge_type] = list()
            edge_type2edge_index[v2u_edge_type].append([
                node_name2idx[v], node_name2idx[u],
            ])
            if edge_type2edge_attr.get(v2u_edge_type) is None:
                edge_type2edge_attr[v2u_edge_type] = list()
            edge_attr = [attr.get('pc_index', 0) / max_pc_index] \
                if attr['type'] == 'cover' else [attr.get('order')]
            edge_type2edge_attr[v2u_edge_type].append(edge_attr)

        # load all data to HeteroData
        for node_type, features in node_type2features.items():
            data[node_type].x = torch.tensor(features)
        for node_type, labels in node_type2labels.items():
            data[node_type].y = torch.tensor(labels)
        for node_type, node_names in node_type2node_name.items():
            data[node_type].position = node_names
        for edge_type, edge_index in edge_type2edge_index.items():
            data[edge_type].edge_index = torch.tensor(edge_index).t().contiguous()
        for edge_type, edge_attr in edge_type2edge_attr.items():
            data[edge_type].edge_attr = torch.tensor(edge_attr)
        return data
