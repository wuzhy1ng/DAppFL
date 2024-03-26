import asyncio
import json
import os
from typing import Dict

from downloaders.defs import JSONRPCDownloader, Downloader
from settings import CACHE_DIR


class TransactionDownloader(Downloader):
    def __init__(self, rpc_url: str):
        self.rpc_url = rpc_url

    async def _preprocess(self, transaction_hash: str, **kwargs):
        return None

    async def _fetch(self, transaction_hash: str, **kwargs):
        return await asyncio.gather(*[
            asyncio.create_task(TraceDownloader(self.rpc_url).download(transaction_hash=transaction_hash)),
            asyncio.create_task(EventLogDownloader(self.rpc_url).download(transaction_hash=transaction_hash)),
        ])

    async def _process(self, result, *args, **kwargs):
        trace, logs = result
        return {
            'trace': trace,
            'logs': logs,
        }


class TraceDownloader(JSONRPCDownloader):
    def get_request_param(self, transaction_hash: str) -> Dict:
        data = {
            "id": 1,
            "jsonrpc": "2.0",
            "params": [
                transaction_hash.lower(),
                {"tracer": "callTracer"},
            ],
            "method": "debug_traceTransaction"
        }
        return {
            "url": self.rpc_url,
            "json": data,
        }

    async def _preprocess(self, transaction_hash: str):
        path = os.path.join(CACHE_DIR, 'trace', '%s.json' % transaction_hash)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return json.load(f)

    async def _process(self, result: str, **kwargs):
        result = json.loads(result)
        result = [item for item in self._parse(result['result'])]

        # cache data
        transaction_hash = kwargs['transaction_hash']
        path = os.path.join(CACHE_DIR, 'trace')
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, '%s.json' % transaction_hash)
        with open(path, 'w') as f:
            json.dump(result, f)
        return result

    def _parse(self, data: dict):
        if not data.get('calls'):
            yield data
            return

        calls = data.pop('calls')
        yield data
        for _call in calls:
            yield from self._parse(_call)


class EventLogDownloader(JSONRPCDownloader):
    def get_request_param(self, transaction_hash: str) -> Dict:
        data = {
            "id": 1,
            "jsonrpc": "2.0",
            "params": [transaction_hash.lower()],
            "method": "eth_getTransactionReceipt"
        }
        return {
            "url": self.rpc_url,
            "json": data,
        }

    async def _preprocess(self, transaction_hash: str):
        path = os.path.join(CACHE_DIR, 'log', '%s.json' % transaction_hash)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return json.load(f)

    async def _process(self, result: str, **kwargs):
        result = json.loads(result)
        result = result["result"]["logs"]

        # cache data
        transaction_hash = kwargs['transaction_hash']
        path = os.path.join(CACHE_DIR, 'log')
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, '%s.json' % transaction_hash)
        with open(path, 'w') as f:
            json.dump(result, f)
        return result


async def test():
    d = TransactionDownloader('https://eth-mainnet.nodereal.io/v1/317f6d43dd4c4acea1fa00515cf02f90')
    rlt = await d.download(transaction_hash='0x54e45ce9b037a6e353284533958147a607ff0569670d62add99d5f5f3b9e09e9')
    print(rlt)


if __name__ == '__main__':
    asyncio.run(test())
