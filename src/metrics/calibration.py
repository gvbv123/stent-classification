import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    ece = 0
    for m in range(len(bin_boundaries) - 1):
        mask = (y_prob > bin_boundaries[m]) & (y_prob <= bin_boundaries[m + 1])
        if np.any(mask):
            bin_prob = y_prob[mask]
            bin_true = y_true[mask]

            bin_conf = np.mean(bin_prob)
            bin_acc = np.mean(bin_true)

            bin_weight = np.sum(mask) / len(y_prob)
            ece += bin_weight * np.abs(bin_acc - bin_conf)

    return ece

def compute_calibration(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    eps = 1e-6
    y_prob_safe = np.clip(y_prob, eps, 1 - eps)
    logits = np.log(y_prob_safe) - np.log(1 - y_prob_safe)
    logits = logits.reshape(-1, 1)

    reg = LogisticRegression(solver="lbfgs")
    reg.fit(logits, y_true)
    slope = reg.coef_[0][0]
    intercept = reg.intercept_[0]

    return {
        "brier": brier,
        "ece": ece,
        "slope": slope,
        "intercept": intercept,
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist()
    }
