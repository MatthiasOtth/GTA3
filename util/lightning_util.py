import torch
import lightning as L
from lightning.pytorch.callbacks import Callback


class StopOnLrCallback(Callback):
    """ Stop training when lr reaches a threshold.
    by default checks after validation. can also check after training.
    """
    def __init__(self, lr_threshold=1e-6, on_train=None, on_val=None):
        super().__init__()
        self.lr_threshold = lr_threshold
        if on_train is None and on_val is None:
            on_val = True  # default to on_val
        self.on_train = on_train or False
        self.on_val = on_val or False

    def _check_should_stop(self, trainer: L.Trainer, pl_module):
        lr = trainer.optimizers[0].param_groups[0]['lr']
        if lr <= self.lr_threshold:
            print(f"Stopping training because lr {lr} <= {self.lr_threshold}")
            trainer.should_stop = True

    def on_train_end(self, trainer, pl_module):
        if self.on_train:
            self._check_should_stop(trainer, pl_module)
    
    def on_validation_end(self, trainer, pl_module):
        if self.on_val:
            self._check_should_stop(trainer, pl_module)

class StopOnValAcc(Callback):
    """ Stop training when lr reaches a threshold.
    by default checks after validation. can also check after training.
    """
    def __init__(self, acc_thresh=1.0, patience=10, on_train=None, on_val=None):
        super().__init__()
        self.acc_thresh = acc_thresh
        if on_train is None and on_val is None:
            on_val = True  # default to on_val
        self.on_train = on_train or False
        self.on_val = on_val or False
        self.counter = 0
        self.patience = patience

    def _check_should_stop(self, trainer: L.Trainer, pl_module):
        val_acc = trainer.callback_metrics["valid_accuracy"]
        if val_acc >= self.acc_thresh:
            self.counter += 1
            if self.counter > self.patience:
                print(f"Stopping training because acc {val_acc} >= {self.acc_thresh}")
                trainer.should_stop = True
        else:
            self.counter = 0

    def on_train_end(self, trainer, pl_module):
        if self.on_train:
            self._check_should_stop(trainer, pl_module)
    
    def on_validation_end(self, trainer, pl_module):
        if self.on_val:
            self._check_should_stop(trainer, pl_module)
