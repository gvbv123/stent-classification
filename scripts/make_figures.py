# scripts/make_figures.py
import os
import json
import glob
import argparse
from typing import Optional, Dict

import numpy as np
import pandas as pd

from src.plots.plot_roc_pr import plot_roc_pr
from src.plots.plot_confusion import plot_confusion
from src.plots.plot_calibration import plot_calibration
from src.plots.plot_train_curves import plot_train_curves


# ---------- helpers ----------

def _load_metrics(path: Optional[str]) -> Optional[Dict]:
    """Optional: Load metrics (json or csv). """
    if not path:
        return None
    if not os.path.exists(path):
        print(f"[make_figures] Metrics path does not exist: {path}")
        return None

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif ext == ".csv":
            row = pd.read_csv(path).iloc[0].to_dict()
            # Convert all values to native types
            return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        else:
            print(f"[make_figures] Unrecognized metrics format: {ext}")
    except Exception as e:
        print(f"[make_figures] Failed to read metrics: {e}")
    return None


def _compute_calibration_from_preds(y_true: np.ndarray,
                                    y_prob: np.ndarray,
                                    n_bins: int = 10) -> Dict:    
    try:
        from sklearn.calibration import calibration_curve
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        return {"prob_true": frac_pos.tolist(), "prob_pred": mean_pred.tolist()}
    except Exception:
        # Fallback: uniform binning
        order = np.argsort(y_prob)
        y_true_sorted = y_true[order]
        y_prob_sorted = y_prob[order]
        idx_splits = np.array_split(np.arange(len(y_prob_sorted)), n_bins)

        prob_true, prob_pred = [], []
        for idx in idx_splits:
            if len(idx) == 0:
                continue
            prob_pred.append(float(np.mean(y_prob_sorted[idx])))
            prob_true.append(float(np.mean(y_true_sorted[idx])))
        return {"prob_true": prob_true, "prob_pred": prob_pred}


def _resolve_log_csv(log_csv: Optional[str],
                     log_dir: Optional[str]) -> Optional[str]:
    """
    Return a valid log.csv to plot training curves:
    - If log_csv is provided and exists, use it directly;
    - If log_dir is provided, merge log*.csv files into a temporary file;
    - If neither is available, return None.
    """
    if log_csv and os.path.exists(log_csv):
        return log_csv

    if log_dir and os.path.isdir(log_dir):
        files = sorted(glob.glob(os.path.join(log_dir, "log*.csv")))
        if not files:
            print(f"[make_figures] No log*.csv found in {log_dir}, skipping training curves.")
            return None

        dfs = []
        for i, f in enumerate(files):
            try:
                df = pd.read_csv(f)
                if "fold" not in df.columns:
                    df["fold"] = i
                dfs.append(df)
            except Exception as e:
                print(f"[make_figures] Failed to read {f}: {e}")

        if not dfs:
            return None

        merged = pd.concat(dfs, ignore_index=True)
        tmp_path = os.path.join(log_dir, "_merged_log.csv")
        merged.to_csv(tmp_path, index=False)
        return tmp_path

    return None


# ---------- main ----------

def main(pred_csv: str,
         out_dir: str,
         metrics_path: Optional[str] = None,
         calib_json: Optional[str] = None,
         log_csv: Optional[str] = None,
         log_dir: Optional[str] = None) -> None:

    os.makedirs(out_dir, exist_ok=True)

    # 1) Read predictions
    df = pd.read_csv(pred_csv)
    if not {"label", "prob", "pred"}.issubset(df.columns):
        raise ValueError("predictions.csv must include columns: label, prob, pred")
    y_true = df["label"].values
    y_prob = df["prob"].values
    y_pred = df["pred"].values

    # 2) ROC / PR
    plot_roc_pr(y_true, y_prob, out_dir, prefix="test")

    # 3) Confusion matrix
    plot_confusion(y_true, y_pred, out_dir, prefix="test")

    # 4) Calibration curve: prefer using calib_json, otherwise compute on the fly
    cal = None
    if calib_json and os.path.exists(calib_json):
        try:
            with open(calib_json, "r", encoding="utf-8") as f:
                cal = json.load(f)
        except Exception as e:
            print(f"[make_figures] Failed to read calib_json, will compute on the fly: {e}")

    if cal is None:
        cal = _compute_calibration_from_preds(y_true, y_prob, n_bins=10)
    plot_calibration(cal["prob_true"], cal["prob_pred"], out_dir, prefix="test")

    # 5) Training curves: log_csv or log_dir (one of the two)
    log_use = _resolve_log_csv(log_csv, log_dir)
    if log_use and os.path.exists(log_use):
        plot_train_curves(log_use, out_dir)
    else:
        print("[make_figures] No available training log provided, skipping training curves.")

    # 6) Optional: Load metrics (currently not directly used)
    _ = _load_metrics(metrics_path)

    print(f"[make_figures] All figures saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True, help="Path to predictions.csv")
    parser.add_argument("--out_dir", required=True, help="Directory to save the figures")

    # Optional parameters
    parser.add_argument("--metrics", required=False,
                        help="Optional: path to metrics.json or metrics.csv")
    parser.add_argument("--calib_json", required=False,
                        help="Optional: path to calibration json; if not provided, will compute from predictions")
    parser.add_argument("--log_csv", required=False,
                        help="Optional: path to single log csv (e.g., runs/log.csv)")
    parser.add_argument("--log_dir", required=False,
                        help="Optional: log directory (containing log_fold*.csv files, will merge them automatically)")

    args = parser.parse_args()
    main(pred_csv=args.pred_csv,
         out_dir=args.out_dir,
         metrics_path=args.metrics,
         calib_json=args.calib_json,
         log_csv=args.log_csv,
         log_dir=args.log_dir)
