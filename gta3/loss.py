import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as L

def L1Loss_L1Alpha(y_pred, y_true, alpha, alpha_weight):
    return F.l1_loss(y_pred, y_true) + alpha_weight * L.norm(alpha.view(-1), ord=1)

def L1Loss_L2Alpha(y_pred, y_true, alpha, alpha_weight):
    return F.l1_loss(y_pred, y_true) + alpha_weight * L.norm(alpha.view(-1), ord=2)

class AlphaRegularizationWrapper(nn.Module):
    def __init__(self, loss: nn.Module, alpha_weight: float):
        super().__init__()
        self.loss = loss
        self.alpha_weight = alpha_weight
    
    def forward(self, y_pred, y_true, alpha=None):
        reg = torch.sigmoid(alpha.view(-1)).sum() if alpha is not None else 0
        return self.loss(y_pred, y_true) + self.alpha_weight * reg
    