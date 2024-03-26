import datetime
from typing import Dict

import aiohttp


class Downloader:
    async def download(self, *args, **kwargs):
        result = await self._preprocess(*args, **kwargs)
        if result is not None:
            return result
        result = await self._fetch(*args, **kwargs)
        return await self._process(result, **kwargs)

    async def _preprocess(self, *args, **kwargs):
        raise NotImplemented()

    async def _fetch(self, *args, **kwargs):
        raise NotImplemented()

    async def _process(self, result, *args, **kwargs):
        raise NotImplemented()


class JSONRPCDownloader(Downloader):
    def __init__(self, rpc_url: str):
        self.rpc_url = rpc_url

    def get_request_param(self, *args, **kwargs) -> Dict:
        raise NotImplemented()

    async def _fetch(self, *args, **kwargs):
        params = self.get_request_param(*args, **kwargs)
        client = aiohttp.ClientSession()
        async with client.post(**params) as response:
            rlt = await response.text()
        await client.close()
        return rlt


class EtherscanDownloader(Downloader):
    def __init__(self, apikey: str):
        self.apikey = apikey

    def get_request_param(self, *args, **kwargs) -> Dict:
        raise NotImplemented()

    async def _fetch(self, *args, **kwargs):
        params = self.get_request_param(*args, **kwargs)
        client = aiohttp.ClientSession()
        async with client.get(**params) as response:
            rlt = await response.text()
        await client.close()
        return rlt
