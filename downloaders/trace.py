import json
import os
from typing import Dict

from downloaders.defs import JSONRPCDownloader
from settings import PC_TRACER, CACHE_DIR


class PCTraceDownloader(JSONRPCDownloader):
    def get_request_param(self, transaction_hash: str) -> Dict:
        data = {
            "id": 1,
            "jsonrpc": "2.0",
            "params": [
                transaction_hash.lower(),
                {"tracer": PC_TRACER},
            ],
            "method": "debug_traceTransaction"
        }
        return {
            "url": self.rpc_url,
            "json": data,
        }

    async def _preprocess(self, transaction_hash: str):
        path = os.path.join(CACHE_DIR, 'pc', '%s.json' % transaction_hash)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return json.load(f)

    async def _process(self, result: str, **kwargs):
        result = json.loads(result)
        result = [item for item in result["result"]]

        # cache data
        transaction_hash = kwargs['transaction_hash']
        path = os.path.join(CACHE_DIR, 'pc')
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, '%s.json' % transaction_hash)
        with open(path, 'w') as f:
            json.dump(result, f)
        return result
