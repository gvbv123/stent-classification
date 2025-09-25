import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from typing import Tuple

# ---- DeLong implementation ----
import scipy.stats as st

def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def _fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    for r in range(k):
        tx[r] = _compute_midrank(positive_examples[r])
        ty[r] = _compute_midrank(negative_examples[r])
    tz = _compute_midrank(predictions_sorted_transposed[0])

    aucs = tz[:m].sum() / (m * n) - (tz[m:].sum()) / (m * n)
    v01 = (tx / n).sum(axis=1) - tz[:m]
    v10 = tz[m:] - (ty / m).sum(axis=1)
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def delong_ci(y_true, y_prob, alpha=0.95) -> Tuple[float, Tuple[float, float]]:
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    order = np.argsort(-y_prob)
    label_1_count = np.sum(y_true)
    predictions_sorted_transposed = np.vstack((y_prob,))[..., order]
    auc, var = _fastDeLong(predictions_sorted_transposed, int(label_1_count))
    auc = roc_auc_score(y_true, y_prob)
    se = np.sqrt(var)
    z = st.norm.ppf((1 + alpha) / 2)
    ci = (auc - z * se, auc + z * se)
    return auc, ci

# ---- Bootstrap ----
def bootstrap_ci(y_true, y_prob, n_bootstrap=2000, alpha=0.95, random_state=42):
    rng = np.random.RandomState(random_state)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(np.array(y_true)[idx])) < 2:
            continue
        aucs.append(roc_auc_score(np.array(y_true)[idx], np.array(y_prob)[idx]))
    aucs = np.array(aucs)
    lower = np.percentile(aucs, (1 - alpha) / 2 * 100)
    upper = np.percentile(aucs, (1 + alpha) / 2 * 100)
    return (lower, upper)
