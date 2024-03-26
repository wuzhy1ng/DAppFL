import asyncio
import sys
import time


class AsyncItemBucket:
    def __init__(self, items: list, qps: int):
        self.items = items
        self.qps = qps

        self._last_get_time = [0 for _ in range(len(self.items))]
        self._get_interval = 1 / (len(self.items) * qps)
        self._lock = asyncio.Lock()

    async def get(self):
        # get the lock
        await self._lock.acquire()

        # choose a provider
        idx, last_get_time = 0, sys.maxsize
        for _idx, _last_get_time in enumerate(self._last_get_time):
            if _last_get_time < last_get_time:
                last_get_time = _last_get_time
                idx = _idx

        # get item
        now = time.time()
        duration = now - last_get_time
        if duration < self._get_interval:
            await asyncio.sleep(self._get_interval - duration)
        self._last_get_time[idx] = time.time()
        item = self.items[idx]

        # release lock and return
        self._lock.release()
        return item


async def test(b):
    print(await b.get())


if __name__ == '__main__':
    b = AsyncItemBucket(items=[1, 2], qps=1)
    tasks = [test(b) for _ in range(10)]
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))
