import torch
from tqdm import tqdm
from src.metrics.basic_metrics import compute_basic_metrics

@torch.no_grad()
def evaluate(model, loader, device):
    """
    在验证集上评估，返回指标 dict
    """
    model.eval()
    y_true, y_prob = [], []

    for x, y, _ in tqdm(loader, desc="Valid"):
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits[:, 1] if logits.ndim == 2 else logits).cpu().numpy()
        y_true.extend(y.numpy())
        y_prob.extend(prob)

    metrics = compute_basic_metrics(y_true, y_prob, prefix="val_")
    return metrics
