import datetime
import math
from typing import Tuple, List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from dataset.pyg import FL4SCDataset
from models.HGT import HGT
from settings import JSONRPCS, SCAN_APIKEYS


def plot_scatter(data_path: str):
    x, y, label = full_embedding(data_path)
    fig, ax = plt.subplots()

    # plot normal point
    ax.scatter(
        x=[_x for i, _x in enumerate(x) if label[i] == 0],
        y=[_y for i, _y in enumerate(y) if label[i] == 0],
        c='b',
    )

    # plot abnormal point
    ax.scatter(
        x=[_x for i, _x in enumerate(x) if label[i] == 1],
        y=[_y for i, _y in enumerate(y) if label[i] == 1],
        c='r',
    )

    plt.show()
    print('ok')


def full_embedding(data_path: str) -> Tuple[List, List, List]:
    # build dataset
    dataset = FL4SCDataset(
        root=data_path,
        net2rpc=JSONRPCS,
        net2apikey=SCAN_APIKEYS,
        leakage=True,
    )

    # split dataset
    train_dataset, test_dataset = random_split(
        dataset,
        lengths=[int(0.7 * len(dataset)), math.ceil(0.3 * len(dataset))],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        pin_memory=True,
    )

    # init model
    model = HGT(**{
        "out_channels": 2,
        "hidden_channels": 32,
        "metadata": dataset.metadata,
        "num_layers": 4,
    })
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # start training
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_data in train_loader:
            optimizer.zero_grad()
            pred = model(
                x_dict={
                    t: torch.as_tensor(batch_data[t].x, dtype=torch.float)
                    for t in batch_data.node_types
                },
                edge_index_dict={
                    t: torch.as_tensor(batch_data[t].edge_index, dtype=torch.long)
                    for t in batch_data.edge_types
                },
            )
            mask = batch_data['FunctionDefinition'].x[:, 2] > 0
            pred = pred['FunctionDefinition'][mask]
            pred = F.log_softmax(pred, dim=1)
            loss = F.nll_loss(pred, batch_data['FunctionDefinition'].y[mask])
            loss.backward()
            total_loss += loss
            optimizer.step()

        print(datetime.datetime.now(), total_loss, epoch + 1)

    # test
    preds, targets = list(), list()
    with torch.no_grad():
        for batch_data in test_loader:
            pred = model(
                x_dict={
                    t: torch.as_tensor(batch_data[t].x, dtype=torch.float)
                    for t in batch_data.node_types
                },
                edge_index_dict={
                    t: torch.as_tensor(batch_data[t].edge_index, dtype=torch.long)
                    for t in batch_data.edge_types
                },
            )
            mask = batch_data['FunctionDefinition'].x[:, 2] > 0
            pred = F.softmax(pred['FunctionDefinition'][mask], dim=1)
            preds.append(pred[:, 1])
            targets.append(batch_data['FunctionDefinition'].y[mask])

    # transform to curve
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    return preds[:, 0].tolist(), preds[:, 1].tolist(), targets.tolist()


if __name__ == '__main__':
    plot_scatter(r'H:\python_projects\FL4SC\data')
