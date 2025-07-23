import torch

def pearson_corr_loss(pred, target, eps=1e-8):
    pred_centered = pred - pred.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=-1)
    denominator = torch.sqrt((pred_centered ** 2).sum(dim=-1) * (target_centered ** 2).sum(dim=-1)) + eps
    corr = numerator / denominator
    return 1 - corr.mean()