import torch
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    余弦退火带 warmup 的 scheduler
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        cur_epoch = self.last_epoch
        if cur_epoch < self.warmup_epochs:
            # warmup: 线性增加
            return [base_lr * (cur_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # cosine 退火
            progress = (cur_epoch - self.warmup_epochs) / float(self.max_epochs - self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


def get_scheduler(cfg, optimizer):
    """
    根据配置创建调度器
    cfg["OPTIM"]["SCHEDULER"]: "cosine" / "steplr"
    cfg["TRAIN"]["EPOCHS"]
    cfg["OPTIM"]["WARMUP_EPOCHS"]
    cfg["OPTIM"]["STEPLR_STEP"], cfg["OPTIM"]["STEPLR_GAMMA"]
    """
    sched_name = cfg["OPTIM"]["SCHEDULER"].lower()
    max_epochs = int(cfg["TRAIN"]["EPOCHS"])

    if sched_name == "cosine":
        warmup_epochs = int(cfg["OPTIM"].get("WARMUP_EPOCHS", 0))
        return WarmupCosineLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            min_lr=1e-6
        )
    elif sched_name == "steplr":
        step = int(cfg["OPTIM"].get("STEPLR_STEP", 10))
        gamma = float(cfg["OPTIM"].get("STEPLR_GAMMA", 0.1))
        return StepLR(optimizer, step_size=step, gamma=gamma)
    else:
        raise ValueError(f"未知调度器: {sched_name}")
