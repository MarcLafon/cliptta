from typing import Union
import os
from argparse import Namespace

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ttavlm.lib import DictAverage


def save_checkpoint(
    save_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    scaler: Union[GradScaler, None],
    train_meter: DictAverage,
    args: Namespace,
) -> None:
    state = {
        "epoch": epoch,
        "train_meter": {k: v.avg for k, v in train_meter.items()},
        "state_dict": model.state_dict()
        if not hasattr(model, "trainable_state_dict")
        else model.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "args": args,
    }

    save_dir = os.path.join(save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # running checkpoint
    model_name = (
        f"model_seed{args.seed}.ckpt" if args.seed is not None else "model.ckpt"
    )
    torch.save(state, os.path.join(save_dir, model_name))

    # epoch checkpoint
    if ((epoch % args.save_freq == 0) and (epoch > 0)) or (epoch + 1 == args.n_epochs):
        model_name = (
            f"model_seed{args.seed}_epoch{epoch}.ckpt"
            if args.seed is not None
            else f"model_epoch{epoch}.ckpt"
        )
        torch.save(state, os.path.join(save_dir, model_name))
