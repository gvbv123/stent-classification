import torch
import torch.nn as nn

def compute_pos_weight(labels):
    """
    根据标签计算 pos_weight = N_neg / N_pos
    Args:
        labels: 1D Tensor 或 list，0/1 标签
    Returns:
        torch.Tensor([pos_weight])
    """
    labels = torch.tensor(labels)
    pos = (labels == 1).sum().item()
    neg = (labels == 0).sum().item()
    if pos == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    return torch.tensor([neg / max(pos, 1)], dtype=torch.float32)


class BCEWithPosWeight(nn.Module):
    """
    BCEWithLogitsLoss，支持 pos_weight
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        """
        Args:
            logits: (N,2) 或 (N,) 原始logits
            targets: (N,) int64 [0,1]
        """
        if logits.ndim == 2 and logits.shape[1] == 2:
            # 取正类logit
            logits = logits[:, 1]
        targets = targets.float()
        return self.criterion(logits, targets)
