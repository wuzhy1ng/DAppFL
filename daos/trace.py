from typing import List

from downloaders.defs import Downloader


class PCTraceItem:
    def __init__(
            self, transaction_hash: str, index: int, pc: int,
            opcode: str, depth: int, address: str, is_error: bool
    ):
        self.transaction_hash = transaction_hash
        self.index = index
        self.pc = pc
        self.opcode = opcode
        self.depth = depth
        self.address = address
        self.is_error = is_error

    def __str__(self):
        return '{} {} {}({})'.format(self.transaction_hash, self.address, self.pc, self.opcode)


class PCTraceDao:
    def __init__(self, downloader: Downloader):
        self.downloader = downloader

    async def get_pc_list(self, transaction_hash: str) -> List[PCTraceItem]:
        """
        Generate a series of `PCTraceItem` from RPC, with fields as
        `pc`, `op`, `depth`, and `address`.

        :param transaction_hash: hash of the specific transaction.
        :return: A generator.
        """
        items = await self.downloader.download(transaction_hash=transaction_hash)
        return [PCTraceItem(
            transaction_hash=transaction_hash,
            index=i,
            pc=int(item.get('pc', -1)),
            opcode=item.get('op', ''),
            depth=int(item.get('depth', -1)),
            address=item.get('address', ''),
            is_error=item.get('is_error', False),
        ) for i, item in enumerate(items)]
