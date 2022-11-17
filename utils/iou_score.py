import torch

def iou_score(y_hat:torch.Tensor,y:torch.Tensor):
    return torch.sum(y_hat)