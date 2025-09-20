import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression

def compute_calibration(y_true, y_prob, n_bins=10):
    """
    计算校准相关指标
    Returns:
        dict, curve points
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Brier score
    brier = brier_score_loss(y_true, y_prob)

    # 校准曲线 (等分概率段)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    # 校准斜率/截距 (logit 回归)
    eps = 1e-6
    logits = np.log(y_prob + eps) - np.log(1 - y_prob + eps)
    logits = logits.reshape(-1, 1)
    reg = LogisticRegression(solver="lbfgs")
    reg.fit(logits, y_true)
    slope = reg.coef_[0][0]
    intercept = reg.intercept_[0]

    return {
        "brier": brier,
        "slope": slope,
        "intercept": intercept,
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist()
    }
