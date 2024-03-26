import os
import tempfile

from settings import TMP_FILE_DIR


async def wrap_run4tmpfile(data: str, async_func):
    fd, path = tempfile.mkstemp(dir=TMP_FILE_DIR)
    path = path.replace('\\', '/')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(data)
    rlt = await async_func(path)
    try:
        os.remove(path)
    except:
        print('warning: can not remove the file in %s' % path)
    return rlt
