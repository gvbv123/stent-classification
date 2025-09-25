import os
import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd

from src.config.loader import load_config
from src.data.dataset import StentDataset
from src.transforms.aug_pipelines import build_transforms
from src.utils.checkpoint import load_checkpoint
from src.utils.io_paths import make_output_dirs
from src.engine.infer_external import infer_external
from src.metrics.basic_metrics import compute_basic_metrics
from src.metrics.ci_delong_boot import delong_ci, bootstrap_ci
from src.metrics.dca_utils import export_dca_csv
from src.export.exporter import Exporter
from src.models.factory import ModelFactory


def main(cfg_paths):
    cfg = load_config(cfg_paths)

    # 1) Load checkpoint path from configuration
    ckpt_path = cfg["TEST"]["CKPT_PATH"]
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Specified model checkpoint does not exist: {ckpt_path}")

    # 2) Create output directory for "test results" (with timestamp)
    dirs = make_output_dirs(cfg)

    # 3) Dataset / Dataloader
    test_set = StentDataset(
        image_dir=cfg["DATA"]["TEST"]["IMAGE_DIR"],
        mask_dir=cfg["DATA"]["TEST"]["MASK_DIR"],
        label_csv=cfg["DATA"]["TEST"]["LABEL_CSV"],
        input_mode=cfg["INPUT"]["MODE"],
        transform=build_transforms(cfg, split="test"),
        binary_norm=cfg["DATA"]["MASK"]["BINARY_NORM"]
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg["TRAIN"]["BATCH_SIZE"],  
        shuffle=False,
        num_workers=cfg["DATA"].get("NUM_WORKERS", 0),
        pin_memory=True
    )

    # 4) Load model and checkpoint (use fixed ckpt_path instead of a new directory)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModelFactory.build(cfg).to(device)
    model = load_checkpoint(model, ckpt_path, device=device)
    print(f"[TEST] Loaded checkpoint: {ckpt_path}")

    # 5) Inference
    df_preds = infer_external(
        model, test_loader, device,
        out_csv=os.path.join(dirs["outputs"], "predictions.csv")
    )

    # 6) Metrics
    metrics = compute_basic_metrics(
        df_preds["label"].values, df_preds["prob"].values, prefix="test_"
    )

    # 7) Confidence intervals
    auc, ci = delong_ci(df_preds["label"].values, df_preds["prob"].values)
    boot_ci = bootstrap_ci(
        df_preds["label"].values, df_preds["prob"].values,
        n_bootstrap=cfg["STATS"]["BOOTSTRAP_ITERS"]
    )

    # 8) DCA export to CSV (for plotting in R)
    dca_df = export_dca_csv(
        df_preds["label"].values, df_preds["prob"].values,
        os.path.join(dirs["outputs"], "dca_curve.csv"),
        tmin=cfg["DCA"]["THRESH_MIN"],
        tmax=cfg["DCA"]["THRESH_MAX"],
        tstep=cfg["DCA"]["THRESH_STEP"]
    )

    # 9) Export all results
    exp = Exporter(dirs["outputs"])
    exp.save_predictions(df_preds)
    exp.save_metrics(metrics)
    exp.save_ci(auc, ci)
    exp.save_dca(dca_df)
    print(f"[TEST] Results saved to: {dirs['outputs']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True, help="Path to configuration file (including main and sub-configurations)")
    args = parser.parse_args()
    main(args.config)
