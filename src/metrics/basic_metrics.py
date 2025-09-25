import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, f1_score

def compute_basic_metrics(y_true, y_prob, prefix=""):
    """
    Compute basic classification metrics.
    Args:
        y_true: list[int] or np.array
        y_prob: list[float] probabilities for the positive class
    Returns:
        dict
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)

    return {
        f"{prefix}auc": auc,
        f"{prefix}auprc": auprc,
        f"{prefix}acc": acc,
        f"{prefix}f1": f1,
        f"{prefix}sens": sens,
        f"{prefix}spec": spec,
        f"{prefix}tn": tn,
        f"{prefix}fp": fp,
        f"{prefix}fn": fn,
        f"{prefix}tp": tp,
    }
