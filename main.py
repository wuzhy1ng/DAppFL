import argparse
import datetime
import os

from sklearn.model_selection import KFold

from algos.focal_loss import FocalLoss
from dataset.transform import format_data_type

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import torch

import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from dataset.pyg import FL4SCDataset
from models.HGT import HGT
from settings import SCAN_APIKEYS, JSONRPCS
from utils.metrics import plot_metrics


def train(data_path: str, model_args: dict, **kwargs):
    print(model_args)
    print(kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu' if kwargs.get('gpu', False) == 'False' else device

    # build dataset
    dataset = FL4SCDataset(
        root=data_path,
        net2rpc=JSONRPCS,
        net2apikey=SCAN_APIKEYS,
        leakage=True,
        mapping=True,
        grad_func=lambda x: x ** (kwargs.get('p_norm') - 1),
        transform=format_data_type,
    )

    # Configuration options
    # Set fixed random number seed
    torch.manual_seed(42)
    top_k_sum = {1: [], 3: [], 5: []}
    MFR_sum = []
    MAR_sum = []

    k_folds = kwargs.get('k_folds', 5)
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        train_ids = train_ids.tolist()
        test_ids = test_ids.tolist()

        # Define model and optimizer
        model = HGT(**{
            "hidden_channels": kwargs.get('hidden_channels', 32),
            "out_channels": 2,
            "metadata": dataset.metadata,
            "num_layers": kwargs.get('num_layers', 4),
            "num_heads": kwargs.get('num_heads', 1),
            **model_args,
        }).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=kwargs.get('lr', 0.001),
            weight_decay=kwargs.get('weight_decay', 5e-4),
        )
        criterion = FocalLoss(alpha=1, gamma=2)
        model.train()

        # Sample elements randomly from a given list of ids, no replacement.
        print(f'FOLD {fold}')
        print('--------------------------------')
        print('train idx:', train_ids)
        print('test idx:', test_ids)

        # Define data loaders for training and testing data in this fold
        train_loader = DataLoader(
            dataset[train_ids],
            batch_size=kwargs.get('batch_size', 64),
            drop_last=True,
            pin_memory=True,
            shuffle=True,
        )
        for epoch in range(kwargs.get('epoch', 10)):
            total_loss = 0
            for batch_data in train_loader:
                batch_data = batch_data.to(device)
                optimizer.zero_grad()
                mask = batch_data['FunctionDefinition'].mask
                pred = model(batch_data)
                pred = pred['FunctionDefinition'][mask]
                loss = criterion(pred, batch_data['FunctionDefinition'].y[mask])
                total_loss += loss.detach()
                loss.backward()
                optimizer.step()
            print('{}, loss {}, epoch {}'.format(
                datetime.datetime.now(), total_loss, epoch + 1
            ))

        # save model
        path = './model_%d.pth' % fold
        print('save model params to:', path)
        torch.save(model, path)

        model.eval()
        for i, loader in enumerate([
            DataLoader(dataset[train_ids], batch_size=1),
            DataLoader(dataset[test_ids], batch_size=1),
        ]):
            if i > 0:
                print(datetime.datetime.now(), 'start test')
            preds, targets = list(), list()
            with torch.no_grad():
                for batch_data in loader:
                    batch_data = batch_data.to(device)
                    mask = batch_data['FunctionDefinition'].mask
                    pred = model(batch_data)
                    pred = pred['FunctionDefinition'][mask]
                    pred = F.softmax(pred, dim=1)
                    preds.append(pred[:, 1].flatten())
                    targets.append(batch_data['FunctionDefinition'].y[mask])
            if i > 0:
                print(datetime.datetime.now(), 'end test')
                top_k, mfr, mar, mar_ranks = plot_metrics(preds, targets)
                top_k_sum[1].append(top_k[0])
                top_k_sum[3].append(top_k[1])
                top_k_sum[5].append(top_k[2])
                MFR_sum.append(mfr)
                MAR_sum.append(mar)
                for j, case_idx in enumerate(test_ids):
                    print('mar@%s' % dataset.processed_file_names[case_idx], mar_ranks[j])
            else:
                plot_metrics(preds, targets)

    print('------------ result --------------')
    print('Recall@Top-{}: {}'.format(1, sum(top_k_sum[1]) / k_folds))
    print('Recall@Top-{}: {}'.format(3, sum(top_k_sum[3]) / k_folds))
    print('Recall@Top-{}: {}'.format(5, sum(top_k_sum[5]) / k_folds))
    print('MFR: {}'.format(sum(MFR_sum) / k_folds))
    print('MAR: {}'.format(sum(MAR_sum) / k_folds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--gpu', type=str, default=False)
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--p_norm', type=int, default=6)
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        model_args=dict(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers
        ), **{
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'epoch': args.epoch,
            'batch_size': args.batch_size,
            'leakage': args.leakage,
            'mapping': args.mapping,
            'k_folds': args.k_folds,
            'p_norm': args.p_norm,
            'gpu': args.gpu,
        }
    )
