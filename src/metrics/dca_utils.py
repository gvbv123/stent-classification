import numpy as np
import pandas as pd
import os

def compute_net_benefit(y_true, y_prob, thresholds):
    """
    决策曲线：计算每个阈值的 net benefit
    NB = (TP/N) - (FP/N) * (pt/(1-pt))
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    N = len(y_true)
    rows = []
    for pt in thresholds:
        pred = (y_prob >= pt).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        nb = (tp / N) - (fp / N) * (pt / (1 - pt))
        rows.append({"threshold": pt, "net_benefit": nb})
    return rows

def export_dca_csv(y_true, y_prob, out_csv, tmin=0.05, tmax=0.60, tstep=0.01):
    thresholds = np.arange(tmin, tmax + 1e-8, tstep)
    rows = compute_net_benefit(y_true, y_prob, thresholds)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[DCA] exported -> {out_csv}")
    return df
