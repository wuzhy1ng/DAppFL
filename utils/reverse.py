from typing import Dict, List, Tuple

from daos.contract import ContractDao
from utils.solc import SourceMappingItem


class SourceMapping2AST:
    def __init__(self, source_mapping: List[SourceMappingItem], ast_nodes: List[Tuple[int, int, int]]):
        self.tabel = self._get_table(source_mapping, ast_nodes)

    def item2ast(self, item: SourceMappingItem) -> Tuple[int, int, int]:
        # Use the mapping table to map item to node
        return self.tabel.get((item.begin, item.offset))

    def _get_table(
            self, source_mapping: List[SourceMappingItem],
            ast_nodes: List[Tuple[int, int, int]],
    ) -> Dict[Tuple[int, int], Tuple[int, int, int]]:
        # establish a mapping table
        table = dict()
        range2node = {(begin, offset): node for node, begin, offset in ast_nodes}
        ranges = [(mapping.begin, mapping.offset) for mapping in source_mapping]
        for r in ranges:
            if not range2node.get(r):
                continue
            table[r] = (range2node[r], r[0], r[1])
        return table


async def test():
    from downloaders.contract import ContractSourceDownloader
    dao = ContractDao(ContractSourceDownloader('YourApiKeyToken'))
    item = await dao.get_compile_item('0xf480ee81a54e21be47aa02d0f9e29985bc7667c4')

    from typed_ast import get_ast_graph
    filename = "contracts/farm/facets/GovernanceFacet/GovernanceFacet.sol"
    graph = await get_ast_graph(item.ast[filename])
    s2a = SourceMapping2AST(
        source_mapping=[m for m in item.source_mapping if m.filename == filename],
        ast_nodes=sorted([
            (node, int(attr['src'].split(':')[0]), int(attr['src'].split(':')[1]))
            for node, attr in graph.nodes(data=True)
        ], key=lambda x: x[1]),
    )

    hits, non_hits = 0, 0
    for m in item.source_mapping:
        if m.filename != filename:
            continue
        ast_node = s2a.item2ast(m)
        if ast_node is not None:
            hits += 1
            continue
        non_hits += 1
    print(hits, non_hits)


if __name__ == '__main__':
    # A unit test must be done here!
    import asyncio

    data = asyncio.run(test())
