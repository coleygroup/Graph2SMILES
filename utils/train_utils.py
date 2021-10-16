import logging
import math
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
from datetime import datetime
from rdkit import RDLogger
from torch.optim.lr_scheduler import _LRScheduler


def param_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def param_norm(m):
    return math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))


def grad_norm(m):
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_logger(args, warning_off: bool = False):
    if warning_off:
        RDLogger.DisableLog("rdApp.*")
    else:
        RDLogger.DisableLog("rdApp.warning")

    os.makedirs(f"./logs/{args.data_name}", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"./logs/{args.data_name}/{args.log_file}.{dt}")
    sh = logging.StreamHandler(sys.stdout)
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def log_tensor(tensor, tensor_name: str):
    logging.info(f"--------------------------{tensor_name}--------------------------")
    logging.info(tensor)
    if isinstance(tensor, torch.Tensor):
        logging.info(tensor.shape)
    elif isinstance(tensor, np.ndarray):
        logging.info(tensor.shape)
    elif isinstance(tensor, list):
        try:
            for item in tensor:
                logging.info(item.shape)
        except Exception as e:
            logging.info(f"Error: {e}")
            logging.info("List items are not tensors, skip shape logging.")


class NoamLR(_LRScheduler):
    """
    Adapted from https://github.com/tugstugi/pytorch-saltnet/blob/master/utils/lr_scheduler.py

    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, model_size, warmup_steps):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        step = max(1, self._step_count)
        scale = self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps**(-1.5))

        return [base_lr * scale for base_lr in self.base_lrs]
