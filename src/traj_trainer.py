import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer


class TrajTrainer(Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def initalize_models(self, model, train_loader, val_loader):
        self.model_snapshot = model
        self.loaders = {"train": train_loader, "val": val_loader}
    
    def per_ex_eval(model, batch):
        input, target = batch
        with torch.no_grad():
            pred = model(input)
        return F.cross_entropy_loss(pred, target, reduction='none')
    
    def on_epoch_start(self, model, train_loader, val_loader):
        """
        Should compute the loss trajectories
        Although on_epoch_start exists, it doesn't provide access to the training or the validation set which is essential for our purposes
        """
        raise NotImplementedError
