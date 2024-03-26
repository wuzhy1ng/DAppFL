import json
from asyncio import subprocess
from typing import Dict

import networkx as nx

from settings import TYPED_AST_CODE, NODE_PATH
from utils.tmpfile import wrap_run4tmpfile


async def get_ast_graph(ast: Dict) -> nx.DiGraph:
    g = nx.DiGraph()
    nodes, edges = await get_typed_ast_items(ast)

    # add node to graph, where each node item is a dict
    # with the attributes of `src`, 'type', and `attr`
    for i, node in enumerate(nodes):
        node_attr = dict(src=node['src'], type=node['type'])
        if len(node.get('attr', dict())) > 0:
            node_attr['attr'] = node['attr']
        nodes[i] = (node['id'], node_attr)
    g.add_nodes_from(nodes)

    # add edges to the graph, where each edge item is a dict
    for i, edge in enumerate(edges):
        edges[i] = (edge['from'], edge['to'], dict(order=edge['order']))
    g.add_edges_from(edges)
    return g


async def get_typed_ast_items(ast: Dict):
    async def _get_typed_ast(path):
        cmd = [NODE_PATH, path]
        # print(cmd)
        process = await subprocess.create_subprocess_shell(
            ' '.join(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output, _ = await process.communicate()
        output = output.decode()
        nodes, edges = list(), list()
        for item in output.split('\n'):
            try:
                item = json.loads(item)
                is_node = item.pop('is_node')
                if is_node:
                    nodes.append(item)
                    continue
                edges.append(item)
            except json.JSONDecodeError:
                continue
        return nodes, edges

    code = TYPED_AST_CODE % json.dumps(ast)
    rlt = await wrap_run4tmpfile(
        data=code,
        async_func=_get_typed_ast
    )
    return rlt
