import os
import argparse
import glob
import torch
import cv2
import pandas as pd
from torch.utils.data import DataLoader
from typing import Optional

from src.config.loader import load_config
from src.data.dataset import StentDataset
from src.transforms.aug_pipelines import build_transforms
from src.utils.checkpoint import load_checkpoint
from src.models.factory import ModelFactory
from src.xai.gradcam import GradCAM, overlay_heatmap
from src.xai.cam_gallery import make_gallery

def _auto_find_last_conv_name(model: torch.nn.Module) -> str:
    last = None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last = name
    if last is None:
        raise RuntimeError("Failed to automatically locate the Conv2d layer. Please specify manually using --target_layer.")
    return last

def _resolve_ckpt_path(cfg: dict, cli_ckpt: Optional[str]) -> str:
    if cli_ckpt:
        return cli_ckpt
    model_cfg = cfg.get("MODEL", {}) or {}
    if model_cfg.get("CKPT_PATH"):
        return model_cfg["CKPT_PATH"]
    
    run_name = cfg.get("OUTPUT", {}).get("RUN_NAME", "run")
    return os.path.join("experiments", run_name, "best.ckpt")

def main(cfg_paths, out_dir, cli_ckpt=None, cli_target_layer=None, nmax=None):
    cfg = load_config(cfg_paths)

    test_set = StentDataset(
        image_dir=cfg["DATA"]["TEST"]["IMAGE_DIR"],
        mask_dir=cfg["DATA"]["TEST"]["MASK_DIR"],
        label_csv=cfg["DATA"]["TEST"]["LABEL_CSV"],
        input_mode=cfg["INPUT"]["MODE"],
        transform=build_transforms(cfg, split="test"),
        binary_norm=cfg["DATA"]["MASK"]["BINARY_NORM"],
    )
    
    if nmax is not None and nmax > 0:
        indices = list(range(min(nmax, len(test_set))))
        from torch.utils.data import Subset
        test_set = Subset(test_set, indices)

    loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = ModelFactory.build(cfg)
    ckpt_path = _resolve_ckpt_path(cfg, cli_ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = load_checkpoint(model, ckpt_path, device=device)
    model = model.to(device).eval()
    print(f"[Load] Checkpoint => {ckpt_path}")

    target_layer = None
    if cli_target_layer:
        target_layer = cli_target_layer
    else:
        xai_map = cfg.get("XAI", {}).get("TARGET_LAYER", {}) or {}
        backbone = cfg.get("MODEL", {}).get("BACKBONE", "")
        target_layer = xai_map.get(backbone)
        if target_layer is None:
            target_layer = _auto_find_last_conv_name(model)
            
    print(f"[GradCAM] Target layer = {target_layer}")
    cam_gen = GradCAM(model, target_layer)

    os.makedirs(out_dir, exist_ok=True)
    per_img_out = os.path.join(out_dir, "per_image")
    os.makedirs(per_img_out, exist_ok=True)

    cams, labels, preds, probs, pids, save_paths = [], [], [], [], [], []

    torch.set_grad_enabled(True)

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x, y, pid = batch
        else:
            x, y, pid = batch["image"], batch["label"], batch["pid"]

        x = x.to(device)
        logits = model(x)

        if logits.ndim == 2 and logits.size(1) >= 2:
            prob = torch.softmax(logits, dim=1)[:, 1].item()
        else:
            prob = torch.sigmoid(logits.reshape(-1)).item()
        pred = int(prob >= 0.5)

        cam = cam_gen.generate(x, class_idx=1)

        img = x[0, 0].detach().cpu().numpy()

        h, w = img.shape[:2]
        if cam.shape[:2] != (h, w):
            cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)

        img_u8 = (img * 255.0).clip(0, 255).astype("uint8")
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

        cam_u8 = (cam * 255.0).clip(0, 255).astype("uint8")
        heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0.0)

        name = str(pid[0]) if isinstance(pid, (list, tuple)) else str(pid)
        name = os.path.splitext(os.path.basename(name))[0]
        save_p = os.path.join(per_img_out, f"{name}_cam.png")
        cv2.imwrite(save_p, overlay)

        cams.append(overlay)
        labels.append(int(y.item()) if hasattr(y, "item") else int(y))
        preds.append(pred)
        probs.append(float(prob))
        pids.append(name)
        save_paths.append(save_p)

    gallery_path = os.path.join(out_dir, "cam_gallery.png")
    make_gallery(cams, labels, preds, probs, pids, gallery_path)
    print(f"[Save] Gallery -> {gallery_path}")

    df = pd.DataFrame({
        "patient_id": pids,
        "label": labels,
        "pred": preds,
        "prob": probs,
        "cam_path": save_paths,
        "ckpt": ckpt_path,
        "target_layer": target_layer,
    })
    csv_path = os.path.join(out_dir, "cams_meta.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"[Save] Metadata -> {csv_path}")
    print(f"[Done] Successfully processed {len(pids)} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--target_layer", default=None)
    parser.add_argument("--nmax", type=int, default=None)
    args = parser.parse_args()

    main(
        cfg_paths=args.config,
        out_dir=args.out_dir,
        cli_ckpt=args.ckpt,
        cli_target_layer=args.target_layer,
        nmax=args.nmax,
    )
