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
    """可选：读取 metrics（json 或 csv）。当前绘图未直接使用，仅为兼容保留。"""
    if not path:
        return None
    if not os.path.exists(path):
        print(f"[make_figures] metrics 路径不存在：{path}")
        return None

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif ext == ".csv":
            row = pd.read_csv(path).iloc[0].to_dict()
            # 统一转为内置类型
            return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        else:
            print(f"[make_figures] 未识别的 metrics 格式：{ext}")
    except Exception as e:
        print(f"[make_figures] 读取 metrics 失败：{e}")
    return None


def _compute_calibration_from_preds(y_true: np.ndarray,
                                    y_prob: np.ndarray,
                                    n_bins: int = 10) -> Dict:
    """当未提供 calib_json 时，现场计算校准曲线点。优先用 sklearn，没有就用简易分箱。"""
    # 尝试 sklearn 更规范的实现
    try:
        from sklearn.calibration import calibration_curve
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        return {"prob_true": frac_pos.tolist(), "prob_pred": mean_pred.tolist()}
    except Exception:
        # 兜底：均匀分箱
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
    返回可用于画训练曲线的 log.csv：
    - 若提供且存在 log_csv，直接用；
    - 否则若提供 log_dir，则合并其中的 log*.csv 成一个临时文件返回；
    - 若都不可用，返回 None。
    """
    if log_csv and os.path.exists(log_csv):
        return log_csv

    if log_dir and os.path.isdir(log_dir):
        files = sorted(glob.glob(os.path.join(log_dir, "log*.csv")))
        if not files:
            print(f"[make_figures] 未在 {log_dir} 下找到 log*.csv，跳过训练曲线。")
            return None

        dfs = []
        for i, f in enumerate(files):
            try:
                df = pd.read_csv(f)
                if "fold" not in df.columns:
                    df["fold"] = i
                dfs.append(df)
            except Exception as e:
                print(f"[make_figures] 读取 {f} 失败：{e}")

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

    # 1) 读取预测
    df = pd.read_csv(pred_csv)
    if not {"label", "prob", "pred"}.issubset(df.columns):
        raise ValueError("predictions.csv 必须包含列：label, prob, pred")
    y_true = df["label"].values
    y_prob = df["prob"].values
    y_pred = df["pred"].values

    # 2) ROC / PR
    plot_roc_pr(y_true, y_prob, out_dir, prefix="test")

    # 3) 混淆矩阵
    plot_confusion(y_true, y_pred, out_dir, prefix="test")

    # 4) 校准曲线：优先用 calib_json，否则现场计算
    cal = None
    if calib_json and os.path.exists(calib_json):
        try:
            with open(calib_json, "r", encoding="utf-8") as f:
                cal = json.load(f)
        except Exception as e:
            print(f"[make_figures] 读取 calib_json 失败，将改为现场计算：{e}")

    if cal is None:
        cal = _compute_calibration_from_preds(y_true, y_prob, n_bins=10)
    plot_calibration(cal["prob_true"], cal["prob_pred"], out_dir, prefix="test")

    # 5) 训练曲线：log_csv 或 log_dir（二选一）
    log_use = _resolve_log_csv(log_csv, log_dir)
    if log_use and os.path.exists(log_use):
        plot_train_curves(log_use, out_dir)
    else:
        print("[make_figures] 未提供可用的训练日志，跳过训练曲线。")

    # 6) 可选：读取 metrics（当前未直接使用）
    _ = _load_metrics(metrics_path)

    print(f"[make_figures] 全部图已输出到：{out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True, help="predictions.csv 路径")
    parser.add_argument("--out_dir", required=True, help="图像输出目录")

    # 兼容的可选参数
    parser.add_argument("--metrics", required=False,
                        help="可选：metrics.json 或 metrics.csv 路径")
    parser.add_argument("--calib_json", required=False,
                        help="可选：calibration json；若不提供，将从 predictions 计算")
    parser.add_argument("--log_csv", required=False,
                        help="可选：单个日志 CSV 路径（如 runs/log.csv）")
    parser.add_argument("--log_dir", required=False,
                        help="可选：日志目录（含 log_fold*.csv 等，将自动合并）")

    args = parser.parse_args()
    main(pred_csv=args.pred_csv,
         out_dir=args.out_dir,
         metrics_path=args.metrics,
         calib_json=args.calib_json,
         log_csv=args.log_csv,
         log_dir=args.log_dir)
