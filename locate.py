import argparse
import asyncio
from typing import List

import torch
import torch.nn.functional as F

from dataset.nx import FL4SCGraph
from dataset.pyg import FL4SCDataset
from dataset.transform import format_data_type
from settings import SCAN_APIKEYS, JSONRPCS
from utils.bucket import AsyncItemBucket


def locate(
        model_path: str, net: str,
        fault_txhash: List[str], faultless_txhash: List[str],
        **kwargs
):
    # load model params
    print('loading model params...')
    model = torch.load(model_path, map_location=torch.device('cpu'))

    # load transaction execution data
    print('collecting transaction executing data...')
    rpc_urls = JSONRPCS.get(net, 'Ethereum')
    apikeys = SCAN_APIKEYS.get(net, 'Ethereum')
    data = FL4SCGraph(
        rpc_bucket=AsyncItemBucket(items=rpc_urls, qps=3),
        apikey_bucket=AsyncItemBucket(items=apikeys, qps=2),
        gamma=kwargs.get('gamma', 0.1),
        epsilon=kwargs.get('epsilon', 1e-3),
        grad_func=lambda _x: _x ** (args.p_norm - 1),
    ).process({
        'fault': {'transaction_hash': [fault_txhash]},
        'faultless': {'transaction_hash': faultless_txhash}
    })
    data = asyncio.get_event_loop().run_until_complete(data)
    data = FL4SCDataset.save_graph2data(data)
    data = format_data_type(data)

    # inference
    print('The following code snippets are most likely to cause faults, '
          'and the ranking is in descending order of faulty suspiciousness:')
    mask = data['FunctionDefinition'].mask
    with torch.no_grad():
        pred = model(data)
    pred = pred['FunctionDefinition'][mask]
    pred = F.softmax(pred, dim=1)
    pred = pred[:, 1].flatten()
    _, indices = torch.topk(pred, kwargs.get('topk', 5))
    positions = list()
    for i, m in enumerate(mask):
        if not m:
            continue
        positions.append(data['FunctionDefinition'].position[i])
    for i, idx in enumerate(indices):
        pos = positions[idx]
        addr, fn, src = pos.split('#')
        begin, offset = src.split(':')
        print('Top-{}: fault function at {} -> {}, offset is {}'.format(
            i, fn, addr, '%s:%s' % (begin, offset)
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fault_txhash', type=str, required=True)
    parser.add_argument('--faultless_txhash', type=str, required=True)
    parser.add_argument('--net', type=str, default='Ethereum')
    parser.add_argument('--model_path', type=str, default='misc/model.pth')
    parser.add_argument('--p_norm', type=int, default=6)
    args = parser.parse_args()

    locate(
        fault_txhash=args.fault_txhash.split(','),
        faultless_txhash=args.faultless_txhash.split(','),
        net=args.net,
        model_path=args.model_path,
        p_norm=args.p_norm,
    )
