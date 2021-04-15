#!/usr/bin/python3

"""Defines training of network."""

import os
import time

from typing import List, Tuple

import numpy as np
import torch
import torch.utils.data as data

from torch import nn

from proj4_code.part2b_patch import gen_patch
from proj4_code.utils import get_disparity, load_model, save_model


use_cuda = True and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
tensor_type = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
torch.set_default_tensor_type(tensor_type)

torch.backends.cudnn.deterministic = True
torch.manual_seed(
    333
)  # do not change this, this is to ensure your result is reproduciable


class AverageMeter(object):
    """Computes and stores the average and current value
    See https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class Trainer:
    """Defines model training."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        batch_size: int,
        ws: int = 11,
        max_epoch: int = 1,
        optimizer: torch.optim = None,
        viz_frequency: int = 1,
        save_frequency: int = 200,
        fname: str = "mc_cnn_network.pth",
        resume_training: bool = False,
        checkpoint_path: str = "train_checkpoint.pth",
    ) -> None:
        """
        Trainer class constructor.

        Args:
            model: MC-CNN network object
            dataloader: torch.utils.data.DataLoader
            batch_size: batch size of the training, for each epoch, there is
                batch_size patches pair as input
            ws: the window size, should be odd number
            max_epoch: number of total training iteration
            optimazer: torch.optim,
            vis_frequency: the frequency to visualize the loss
            save_frequency: the frequency to save the model
            fname: path for saving the trained model
            resume_training: whether to start training from interruption by
                model saved in checkpoint_path
            checkpoint_path: the path for save the checkpoint every epoch
        """

        self.model = model
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.ws = ws
        self.max_epoch = max_epoch
        self.optimizer = optimizer
        self.viz_frequency = viz_frequency
        self.save_frequency = save_frequency
        self.fname = fname
        self.resume_training = resume_training
        self.checkpoint_path = checkpoint_path
        self.train_loss_history = []
        self.train_acc_history = []
        self.epoch = None

        # Resume training
        if self.resume_training and os.path.exists(self.checkpoint_path):
            train_checkpoint = torch.load(self.checkpoint_path)
            if (
                train_checkpoint["fname"] == self.fname
                and os.path.exists(self.fname)
                and train_checkpoint["ws"] == self.ws
            ):
                self.train_loss_history = train_checkpoint["train_loss_history"]
                self.train_acc_history = train_checkpoint["train_acc_history"]
                self.epoch = train_checkpoint["epoch"]
                self.offset = train_checkpoint.get("offset", 0)
                optimizer.load_state_dict(train_checkpoint["optimizer_state"])
                self.model = load_model(
                    self.model, train_checkpoint["fname"], device=device, strict=True
                )

    def run_train_epoch(self, epoch: int) -> None:
        """Runs a training epoch."""

        train_loss_meter = AverageMeter("Average Train Loss within Epoch")

        self.model.train()
        for _, (x_batch_tr, y_batch_tr) in enumerate(self.dataloader):
            n = y_batch_tr.shape[0]
            if torch.cuda.is_available():
                x_batch_tr = x_batch_tr.cuda().to(device)
                y_batch_tr = y_batch_tr.cuda().to(device)
            output = self.model(x_batch_tr)
            self.loss = self.model.criterion(
                output, y_batch_tr.view(self.batch_size, 1)
            )
            batch_train_loss = self.loss.data.cpu().item()
            train_loss_meter.update(val=batch_train_loss, n=n)
            yhat = torch.where(output > 0.5, 1, 0)
            acc = (yhat.view(-1) == y_batch_tr.view(-1)).sum() / self.batch_size
            self.train_acc = acc.cpu().item()
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

        # store the avg loss over the whole training set, from this epoch
        self.train_loss_history.append(train_loss_meter.avg)
        self.train_acc_history.append(self.train_acc)

        # Save checkpoint every epoch
        train_checkpoint = {
            "fname": self.fname,
            "ws": self.ws,
            "train_loss_history": self.train_loss_history,
            "train_acc_history": self.train_acc_history,
            "epoch": epoch,
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(train_checkpoint, self.checkpoint_path)

        # Save model every save_frequency
        if epoch == 1 or ((epoch) % self.save_frequency) == 0:
            save_model(self.model, self.fname)

    def train(
        self, val_dataloader: torch.utils.data.DataLoader
    ) -> Tuple[List[float], List[float]]:
        """Train the model."""
        val_loss_history = []
        val_acc_history = []
        start_epoch = self.epoch if self.epoch is not None else 0
        start_time = time.time()
        # Training epoch
        for epoch in range(start_epoch, self.max_epoch):
            self.run_train_epoch(epoch)
            val_loss, val_acc = self.run_val_epoch(epoch, self.model, val_dataloader)
            val_loss_history += [val_loss]
            val_acc_history += [val_acc]
            # Print loss
            if ((epoch) % self.viz_frequency) == 0:
                print(
                    f"Time elapsed: {(time.time() - start_time)//60:.2f}min {(time.time() - start_time)%60:.2f}s, "
                    + f" Iteration: {epoch+1}/{self.max_epoch}, "
                    + f"Train Loss: {self.loss.data:.4f}, "
                    + f"Train accuracy: {self.train_acc:.2f}, "
                    + f"Val loss:{val_loss:.4f}, "
                    + f"Val accuracy: {val_acc:.2f}"
                )

        return (
            self.train_loss_history,
            self.train_acc_history,
            val_loss_history,
            val_acc_history,
        )

    def run_val_epoch(
        self, epoch: int, model: nn.Module, val_dataloader: torch.utils.data.DataLoader
    ) -> float:
        """Runs validation evaluation on each epoch."""
        x_batch, y_batch = val_dataloader[epoch % len(val_dataloader)]
        acc_meter = AverageMeter("Average validation accuracy")

        with torch.no_grad():
            for _, (x_batch, y_batch) in enumerate(self.dataloader):
                # n is the number of patch pairs
                n = x_batch.shape[0] / 2
                output = model(x_batch)
                loss = model.criterion(output, y_batch.view(self.batch_size, 1))
                yhat = torch.where(output > 0.5, 1, 0)
                batch_acc = (yhat.view(-1) == y_batch.view(-1)).sum() / n
                acc_meter.update(val=batch_acc, n=n)

            return loss.data.cpu().item(), acc_meter.avg
