import numpy as np
import random
import cv2
from typing import Dict, Any, Optional

from .preprocess import (
    percentile_clip, scale_to_01, resize_image_and_mask, DatasetZScore, apply_clahe
)

# For control and lightweight implementation, we use OpenCV + random parameters for common augmentations.
# If you prefer Albumentations, you can replace this file with an Albumentations version.

def random_affine(
    img: np.ndarray,
    mask: Optional[np.ndarray],
    rot_deg: float,
    trans_ratio: float,
    scale_range: tuple
):
    """
    Purely geometric augmentation: rotation/translation/scale, center-aligned; mask uses nearest neighbor
    Parameters are randomly chosen within [-rot_deg, +rot_deg], [-trans_ratio, +trans_ratio], [scale_range]
    """
    h, w = img.shape
    angle = random.uniform(-rot_deg, rot_deg)
    tx = random.uniform(-trans_ratio, trans_ratio) * w
    ty = random.uniform(-trans_ratio, trans_ratio) * h
    scale = random.uniform(scale_range[0], scale_range[1])

    M = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle, scale)
    M[:, 2] += [tx, ty]  # translate

    img2 = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    m2 = None
    if mask is not None:
        m2 = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return img2, m2


def random_brightness_contrast_gamma(
    img: np.ndarray,
    brightness: float,
    contrast: float,
    gamma: float
):
    """
    Brightness/contrast/gamma augmentation, input and output assumed to be in [0,1]
    brightness/contrast/gamma represent the maximum shift magnitude (±)
    """
    # brightness
    if brightness > 0:
        delta = random.uniform(-brightness, brightness)
        img = np.clip(img + delta, 0.0, 1.0)
    # contrast
    if contrast > 0:
        factor = random.uniform(1.0 - contrast, 1.0 + contrast)
        mean = img.mean()
        img = np.clip((img - mean) * factor + mean, 0.0, 1.0)
    # gamma
    if gamma > 0:
        g = random.uniform(1.0 - gamma, 1.0 + gamma)
        img = np.clip(np.power(np.clip(img, 0.0, 1.0), g), 0.0, 1.0)
    return img


def random_motion_blur(img: np.ndarray, prob: float, kmin: int = 3, kmax: int = 7):
    """
    Simple motion blur (random horizontal/vertical), input/output [0,1]
    """
    if random.random() > prob:
        return img
    k = random.randint(kmin, kmax)
    if k % 2 == 0:
        k += 1
    # Randomly choose horizontal or vertical kernel
    kernel = np.zeros((k, k), dtype=np.float32)
    if random.random() < 0.5:
        kernel[k//2, :] = 1.0 / k
    else:
        kernel[:, k//2] = 1.0 / k
    img255 = (img * 255.0).astype(np.uint8)
    blurred = cv2.filter2D(img255, -1, kernel)
    return blurred.astype(np.float32) / 255.0


class TrainTransform:
    """
    Training augmentation pipeline:
      p1–p99 clipping -> /255 -> resize -> (optional) CLAHE(prob) -> geometric/lighting augmentation -> z-score
    All augmentations are applied to the image channel only, mask undergoes only geometric transformations (retain 0/1).
    """
    def __init__(
        self,
        img_size: int,
        clip_percentiles=(1.0, 99.0),
        level: str = "light",
        # light parameters
        rot_deg: float = 7,
        trans_slide: float = 0.05,
        scale_range=(0.95, 1.05),
        brightness: float = 0.10,
        contrast: float = 0.10,
        gamma: float = 0.10,
        motion_blur_prob: float = 0.10,
        clahe_prob: float = 0.20,
        # strong parameters (override when level="strong")
        strong_params: Optional[dict] = None,
        # z-score
        zscore_stats: Optional[dict] = None
    ):
        self.img_size = int(img_size)
        self.clip_percentiles = clip_percentiles
        self.level = level
        self.params = dict(
            rot_deg=rot_deg,
            trans_slide=trans_slide,
            scale_range=scale_range,
            brightness=brightness,
            contrast=contrast,
            gamma=gamma,
            motion_blur_prob=motion_blur_prob,
            clahe_prob=clahe_prob
        )
        if self.level == "strong" and strong_params is not None:
            self.params.update(strong_params)

        self.znorm = DatasetZScore(zscore_stats["mean"], zscore_stats["std"]) if zscore_stats else None

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray], label: int, pid: str) -> Dict[str, Any]:
        # 1) Intensity clipping -> [0,1]
        image = percentile_clip(image, self.clip_percentiles[0], self.clip_percentiles[1])
        image = scale_to_01(image)

        # 2) Resize (first apply geometric consistent resizing, augmentations occur at this size)
        image, mask = resize_image_and_mask(image, mask, self.img_size)

        # 3) CLAHE (probability-based)
        if random.random() < self.params["clahe_prob"]:
            image = apply_clahe(image, clip_limit=2.0, tile_grid_size=8)

        # 4) Geometric augmentation (rotation/translation/scale)
        image, mask = random_affine(
            image, mask,
            rot_deg=self.params["rot_deg"],
            trans_ratio=self.params["trans_slide"],
            scale_range=self.params["scale_range"]
        )

        # 5) Lighting augmentation (brightness/contrast/gamma)
        image = random_brightness_contrast_gamma(
            image,
            brightness=self.params["brightness"],
            contrast=self.params["contrast"],
            gamma=self.params["gamma"]
        )

        # 6) Motion blur (probability-based)
        image = random_motion_blur(image, prob=self.params["motion_blur_prob"], kmin=3, kmax=7)

        # 7) z-score (mask remains unchanged)
        if self.znorm is not None:
            image = self.znorm(image)

        out = {"image": image.astype(np.float32), "mask": None if mask is None else mask.astype(np.float32),
               "label": int(label), "pid": pid}
        return out


class EvalTransform:
    """
    Validation/test set: Deterministic preprocessing (no random augmentations)
      p1–p99 clipping -> /255 -> resize -> z-score
    """
    def __init__(
        self,
        img_size: int,
        clip_percentiles=(1.0, 99.0),
        zscore_stats: Optional[dict] = None
    ):
        self.img_size = int(img_size)
        self.clip_percentiles = clip_percentiles
        self.znorm = DatasetZScore(zscore_stats["mean"], zscore_stats["std"]) if zscore_stats else None

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray], label: int, pid: str) -> Dict[str, Any]:
        image = percentile_clip(image, self.clip_percentiles[0], self.clip_percentiles[1])
        image = scale_to_01(image)
        image, mask = resize_image_and_mask(image, mask, self.img_size)
        if self.znorm is not None:
            image = self.znorm(image)
        out = {"image": image.astype(np.float32), "mask": None if mask is None else mask.astype(np.float32),
               "label": int(label), "pid": pid}
        return out


# ------------------------------
# Factory function: Generate transform based on cfg
# ------------------------------

def build_transforms(cfg, split: str, zscore_stats: Optional[dict] = None):
    """
    Args:
        cfg: Configuration object/dictionary read from YAML
        split: "train" / "val" / "test"
        zscore_stats: {"mean": float, "std": float}, obtained from training, used for val/test
    Returns:
        callable transform(dict) -> dict
    """
    img_size = int(cfg["INPUT"]["IMG_SIZE"])
    p_low, p_high = cfg["PREPROCESS"]["CLIP_PERCENTILES"]

    if split == "train" and cfg["AUG"]["LEVEL"].lower() != "none":
        level = cfg["AUG"]["LEVEL"].lower()
        strong_params = None
        if level == "strong":
            sp = cfg["AUG"].get("STRONG", {})
            strong_params = dict(
                rot_deg=sp.get("ROT_DEG", 10),
                trans_slide=sp.get("TRANS_SLIDE", 0.08),
                scale_range=tuple(sp.get("SCALE_RANGE", [0.90, 1.10])),
                brightness=sp.get("BRIGHTNESS", 0.20),
                contrast=sp.get("CONTRAST", 0.20),
                gamma=sp.get("GAMMA", 0.20),
                motion_blur_prob=sp.get("MOTION_BLUR_PROB", 0.20),
                clahe_prob=sp.get("CLAHE_PROB", 0.30),
            )
        tfm = TrainTransform(
            img_size=img_size,
            clip_percentiles=(p_low, p_high),
            level=level,
            rot_deg=cfg["AUG"]["ROT_DEG"],
            trans_slide=cfg["AUG"]["TRANS_SLIDE"],
            scale_range=tuple(cfg["AUG"]["SCALE_RANGE"]),
            brightness=cfg["AUG"]["BRIGHTNESS"],
            contrast=cfg["AUG"]["CONTRAST"],
            gamma=cfg["AUG"]["GAMMA"],
            motion_blur_prob=cfg["AUG"]["MOTION_BLUR_PROB"],
            clahe_prob=cfg["AUG"]["CLAHE_PROB"],
            strong_params=strong_params,
            zscore_stats=zscore_stats
        )
    else:
        tfm = EvalTransform(
            img_size=img_size,
            clip_percentiles=(p_low, p_high),
            zscore_stats=zscore_stats
        )

    # Wrap for compatibility with dataset.py
    def _wrapper(**data):
        return tfm(**data)

    return _wrapper
