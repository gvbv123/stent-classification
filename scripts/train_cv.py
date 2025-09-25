import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config.loader import load_config
from src.data.dataset import StentDataset
from src.transforms.aug_pipelines import build_transforms
from src.utils.seed import set_seed
from src.utils.io_paths import make_output_dirs
from src.utils.logger import CSVLogger
from src.engine.trainer import Trainer
from src.losses.bce_posweight import compute_pos_weight

def main(cfg_paths):
    cfg = load_config(cfg_paths)
    set_seed(cfg["TRAIN"]["SEED"])
    dirs = make_output_dirs(cfg)

    # 1) Load entire training dataset
    df = pd.read_csv(cfg["DATA"]["TRAIN"]["LABEL_CSV"])
    y_all = df["label"].values
    kfold = StratifiedKFold(n_splits=cfg["DATA"]["SPLITS"]["FOLDS"], shuffle=True, random_state=cfg["TRAIN"]["SEED"])

    fold = 0
    for train_idx, val_idx in kfold.split(df["patient_id"], y_all):
        print(f"\n===== Fold {fold} =====")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # 2) Sample pos_weight
        pos_weight = compute_pos_weight(train_df["label"].values)

        # 3) Dataset
        train_set = StentDataset(
            image_dir=cfg["DATA"]["TRAIN"]["IMAGE_DIR"],
            mask_dir=cfg["DATA"]["TRAIN"]["MASK_DIR"],
            label_csv=cfg["DATA"]["TRAIN"]["LABEL_CSV"],
            input_mode=cfg["INPUT"]["MODE"],
            transform=build_transforms(cfg, split="train"),
            binary_norm=cfg["DATA"]["MASK"]["BINARY_NORM"]
        )
        val_set = StentDataset(
            image_dir=cfg["DATA"]["TRAIN"]["IMAGE_DIR"],
            mask_dir=cfg["DATA"]["TRAIN"]["MASK_DIR"],
            label_csv=cfg["DATA"]["TRAIN"]["LABEL_CSV"],
            input_mode=cfg["INPUT"]["MODE"],
            transform=build_transforms(cfg, split="val"),
            binary_norm=cfg["DATA"]["MASK"]["BINARY_NORM"]
        )

        train_subset = Subset(train_set, train_idx)
        val_subset = Subset(val_set, val_idx)

        train_loader = DataLoader(train_subset, batch_size=cfg["TRAIN"]["BATCH_SIZE"], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=cfg["TRAIN"]["BATCH_SIZE"], shuffle=False)

        # 4) Model
        from src.models.factory import ModelFactory
        model = ModelFactory.build(cfg)

        trainer = Trainer(cfg, model, train_loader, val_loader,
                          device="cuda" if torch.cuda.is_available() else "cpu",
                          pos_weight=pos_weight,
                          out_dir=dirs["runs"])
        history = trainer.train()

        # 5) Save logs
        log_path = os.path.join(dirs["runs"], f"log_fold{fold}.csv")
        pd.DataFrame(history).to_csv(log_path, index=False)

        fold += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True, help="Path to configuration file (including main and sub-configurations)")
    args = parser.parse_args()
    main(args.config)
