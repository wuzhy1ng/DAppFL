import torch
from typing import List


def plot_metrics(preds: List[torch.Tensor], targets: List[torch.Tensor]) -> List:
    top_k = []
    # top-k Recall
    for k in [1, 3, 5]:
        hits = 0
        for pred, target in zip(preds, targets):
            _, indices = torch.topk(pred, k)
            if torch.sum(target[indices]) > 0:
                hits += 1
        print('Recall@Top-{}: {}'.format(k, hits / len(preds)))
        top_k.append(hits / len(preds))

    # MFR
    sum_rank = 0
    for pred, target in zip(preds, targets):
        _, _ranks = torch.sort(pred, descending=True)
        for i, is_fault in enumerate(target[_ranks]):
            if is_fault:
                sum_rank += (i + 1)
                break
    print('MFR: {}'.format(sum_rank / len(preds)))
    MFR = sum_rank / len(preds)

    # MAR
    ranks = list()
    for pred, target in zip(preds, targets):
        _, _ranks = torch.sort(pred, descending=True)
        idx2rank = {int(idx): r for r, idx in enumerate(_ranks)}
        sum_fault_rank = 0
        for idx, is_fault in enumerate(target):
            if not is_fault:
                continue
            sum_fault_rank += (idx2rank[idx] + 1)
        ranks.append(sum_fault_rank / torch.sum(target))
    print('MAR: {}'.format(sum(ranks) / len(preds)))
    MAR = sum(ranks) / len(preds)
    print()
    return [top_k, MFR, MAR, ranks]
