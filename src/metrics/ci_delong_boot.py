import numpy as np
import scipy.stats as st
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from typing import Tuple, List, Union

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

    v01 = tx / n
    v10 = 1.0 - ty / m

    if k == 1:
        sx = np.var(v01, ddof=1)
        sy = np.var(v10, ddof=1)
        delongcov = sx / m + sy / n
    else:
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n

    return delongcov

def delong_test(y_true: np.ndarray, y_prob_a: np.ndarray, y_prob_b: np.ndarray) -> float:
    y_true = np.array(y_true)
    y_prob_a = np.array(y_prob_a)
    y_prob_b = np.array(y_prob_b)

    auc_a = roc_auc_score(y_true, y_prob_a)
    auc_b = roc_auc_score(y_true, y_prob_b)

    order = np.argsort(-y_true)
    y_true_sorted = y_true[order]
    m = int(np.sum(y_true_sorted))

    preds = np.vstack([y_prob_a[order], y_prob_b[order]])

    cov_matrix = _fastDeLong(preds, m)

    var_diff = cov_matrix[0, 0] + cov_matrix[1, 1] - 2 * cov_matrix[0, 1]

    z = (auc_a - auc_b) / np.sqrt(var_diff + 1e-12)
    p_value = 2 * st.norm.sf(np.abs(z))

    return p_value

def delong_ci(y_true, y_prob, alpha=0.95) -> Tuple[float, Tuple[float, float]]:
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    auc = roc_auc_score(y_true, y_prob)

    order = np.argsort(-y_true)
    m = int(np.sum(y_true))
    preds = np.vstack([y_prob[order]])

    var = _fastDeLong(preds, m)
    se = np.sqrt(var)

    z = st.norm.ppf((1 + alpha) / 2)
    ci = (max(0, auc - z * se), min(1, auc + z * se))
    return auc, ci

def bootstrap_ci(y_true, y_prob, n_bootstrap=2000, alpha=0.95, random_state=42):
    rng = np.random.RandomState(random_state)
    aucs = []
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))

    aucs = np.array(aucs)
    lower = np.percentile(aucs, (1 - alpha) / 2 * 100)
    upper = np.percentile(aucs, (1 + alpha) / 2 * 100)
    return (lower, upper)
