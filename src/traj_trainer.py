#!/bin/python

import torch
import torch.nn.functional as F
import numpy as np

from pytorch_lightning import Trainer


class TrajTrainer(Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def initalize_models(self, model, train_loader, val_loader):
        self.model_snapshot = model
        self.loader_snapshot = {"train": train_loader, "val": val_loader}
        self.model_snapshot.train_loss_trajectories = None
        self.model_snapshot.probe_loss_trajectories = None
        self.per_ex_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    def compute_loss_trajectories(self):
        # TODO: Convert validation set into actual probes -- mislabeled + OOD (edge maps)
        
        num_train_ex = len(self.loader_snapshot["train"].dataset)
        num_val_ex = len(self.loader_snapshot["val"].dataset)
        print(f"Num training examples: {num_train_ex} / Num validation examples: {num_val_ex}")
        train_loss_vals = [None for _ in range(num_train_ex)]
        val_loss_vals = [None for _ in range(num_val_ex)]
        for set_idx, set in enumerate(["train", "val"]):
            for batch in self.loader_snapshot[set]:
                global_index, data, target = batch
                output = self.model_snapshot(data)
                loss_vals = self.per_ex_criterion(output, target)
                if set_idx == 0:  # Train set
                    for local_idx, global_idx in enumerate(global_index):
                        assert train_loss_vals[global_idx] is None
                        train_loss_vals[global_idx] = loss_vals[local_idx]
                else:  # Validation set
                    for local_idx, global_idx in enumerate(global_index):
                        assert val_loss_vals[global_idx] is None
                        val_loss_vals[global_idx] = loss_vals[local_idx]
        
        assert not any([x is None for x in train_loss_vals])
        assert not any([x is None for x in val_loss_vals])
        
        # Append the loss values in the loss trajectory
        if self.model_snapshot.train_loss_trajectories is None:
            assert self.model_snapshot.probe_loss_trajectories is None
            # Reshape done for concatenation
            self.model_snapshot.train_loss_trajectories = np.array(train_loss_vals).reshape(-1, 1)
            self.model_snapshot.probe_loss_trajectories = np.array(val_loss_vals).reshape(-1, 1)
        else:
            self.model_snapshot.train_loss_trajectories = np.concatenate([self.model_snapshot.train_loss_trajectories, train_loss_vals], axis=1)
            self.model_snapshot.probe_loss_trajectories = np.concatenate([self.model_snapshot.probe_loss_trajectories, val_loss_vals], axis=1)
    
    def on_epoch_start(self, model, train_loader, val_loader):
        """
        Should compute the loss trajectories
        Although on_epoch_start exists, it doesn't provide access to the training or the validation set which is essential for our purposes
        """
        raise NotImplementedError
