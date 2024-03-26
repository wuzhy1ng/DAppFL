import argparse

import torch

from dataset.pyg import FL4SCDataset
from dataset.transform import format_data_type

from settings import SCAN_APIKEYS, JSONRPCS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--p_norm', type=int, default=6)
    args = parser.parse_args()
    dataset = FL4SCDataset(
        root=args.data_path,
        net2rpc=JSONRPCS,
        net2apikey=SCAN_APIKEYS,
        grad_func=lambda _x: _x ** (args.p_norm - 1),
        mapping=False,
        transform=format_data_type,
    )

    # scan the fault element number
    print('Fault elements:')
    for i, d in enumerate(dataset):
        y_label = d[dataset.target_element_type].y.type(torch.bool)
        print(
            dataset.processed_file_names[i],
            torch.sum(d[dataset.target_element_type].y),
            torch.sum(d[dataset.target_element_type].x[y_label][:, 1])
        )
        position = d[dataset.target_element_type].position
        print([pos for i, pos in enumerate(position) if y_label[i]])
        print()
    print()

    # scan the leakage dist
    print('Money leakage total:')
    for i, d in enumerate(dataset):
        sum_leakage = d[dataset.target_element_type].x[:, -1]
        sum_leakage = torch.sum(sum_leakage)
        y = d[dataset.target_element_type].y
        target_leakage = torch.sum(d[dataset.target_element_type].x[:, -1][y])
        print(dataset.processed_file_names[i], sum_leakage, target_leakage)
    print()

    # scan the case quality
    print('Case cover faults:')
    for i, d in enumerate(dataset):
        te = d[dataset.target_element_type]
        y = te.y[te.x[:, 1] > 0]
        print(dataset.processed_file_names[i], torch.sum(y))
    print("ok")

    # statistic
    num_tgts, num_faulty_tgts = 0, 0
    for d in dataset:
        num_tgts += d[dataset.target_element_type].num_nodes
        num_faulty_tgts += torch.sum(d[dataset.target_element_type].y)
    print('#faulty elements / #target elements: {} / {}'.format(
        num_faulty_tgts, num_tgts
    ))
