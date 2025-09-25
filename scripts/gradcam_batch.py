import os
import argparse
import torch
import cv2
import pandas as pd
from torch.utils.data import DataLoader

from src.config.loader import load_config
from src.data.dataset import StentDataset
from src.transforms.aug_pipelines import build_transforms
from src.utils.checkpoint import load_checkpoint
from src.models.factory import ModelFactory
from src.xai.gradcam import GradCAM, overlay_heatmap
from src.xai.cam_gallery import make_gallery

def main(cfg_paths, out_dir):
    cfg = load_config(cfg_paths)

    # Dataset (Test set)
    test_set = StentDataset(
        image_dir=cfg["DATA"]["TEST"]["IMAGE_DIR"],
        mask_dir=cfg["DATA"]["TEST"]["MASK_DIR"],
        label_csv=cfg["DATA"]["TEST"]["LABEL_CSV"],
        input_mode=cfg["INPUT"]["MODE"],
        transform=build_transforms(cfg, split="test"),
        binary_norm=cfg["DATA"]["MASK"]["BINARY_NORM"]
    )
    loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Model
    model = ModelFactory.build(cfg)
    ckpt = os.path.join("experiments", cfg["OUTPUT"]["RUN_NAME"], "best.ckpt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(model, ckpt, device=device)
    model = model.to(device)

    target_layer = cfg["XAI"]["TARGET_LAYER"][cfg["MODEL"]["BACKBONE"]]
    cam_gen = GradCAM(model, target_layer)

    cam_list, labels, preds, probs, pids = [], [], [], [], []
    for x, y, pid in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits[:, 1] if logits.ndim == 2 else logits).cpu().item()
        pred = int(prob >= 0.5)
        cam = cam_gen.generate(x, class_idx=1)
        img = x[0,0].cpu().numpy()
        overlay = overlay_heatmap(img, cam)

        cam_list.append(overlay)
        labels.append(int(y.item()))
        preds.append(pred)
        probs.append(prob)
        pids.append(pid[0])

    os.makedirs(out_dir, exist_ok=True)
    make_gallery(cam_list, labels, preds, probs, pids, os.path.join(out_dir, "cam_gallery.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args.config, args.out_dir)
