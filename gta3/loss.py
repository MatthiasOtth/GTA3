import torch.nn.functional as F
import torch.linalg as L

def L1Loss_L1Alpha(y_pred, y_true, alpha, alpha_weight):
    return F.l1_loss(y_pred, y_true) + alpha_weight * L.norm(alpha.view(-1), ord=1)

def L1Loss_L2Alpha(y_pred, y_true, alpha, alpha_weight):
    return F.l1_loss(y_pred, y_true) + alpha_weight * L.norm(alpha.view(-1), ord=2)