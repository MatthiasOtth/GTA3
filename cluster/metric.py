import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def accuracy_SBM(scores, targets):
    """
    see: https://github.com/graphdeeplearning/benchmarking-gnns/blob/b6c407712fa576e9699555e1e035d1e327ccae6c/train/metrics.py#L34C1-L51C15
    """
    S = targets.cpu().numpy()
    C = np.argmax( torch.nn.Softmax(dim=1)(scores).cpu().detach().numpy() , axis=1 )
    CM = confusion_matrix(S,C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets==r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r,r]/ float(cluster.shape[0])
            if CM[r,r]>0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100.* np.sum(pr_classes)/ float(nb_classes)
    return acc

def accuracy_SBM_fixed(scores, targets):
    """
    see: https://github.com/graphdeeplearning/benchmarking-gnns/blob/b6c407712fa576e9699555e1e035d1e327ccae6c/train/metrics.py#L34C1-L51C15
    """
    S = targets.cpu().numpy()
    C = np.argmax( torch.nn.Softmax(dim=1)(scores).cpu().detach().numpy() , axis=1 )
    CM = confusion_matrix(S,C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets==r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r,r]/ float(cluster.shape[0])
            nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100.* np.sum(pr_classes)/ float(nb_non_empty_classes)
    return acc