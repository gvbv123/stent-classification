import numpy as np
import random
import cv2
from typing import Dict, Any, Optional

from .preprocess import (
    percentile_clip, scale_to_01, resize_image_and_mask, DatasetZScore, apply_clahe
)

# 为了可控与轻量，这里用 OpenCV + 随机参数实现常用增强
# 如果你 prefer Albumentations，可把本文件替换为 Albumentations 版（我也能给你）。

def random_affine(
    img: np.ndarray,
    mask: Optional[np.ndarray],
    rot_deg: float,
    trans_ratio: float,
    scale_range: tuple
):
    """
    纯几何增强：旋转/平移/缩放，中心对齐；mask 使用最近邻
    参数在 [-rot_deg, +rot_deg]、[-trans_ratio, +trans_ratio]、[scale_range] 内随机
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
    亮度/对比度/Gamma 增强，输入与输出假定在 [0,1]
    brightness/contrast/gamma 表示最大偏移幅度（±）
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
    简易运动模糊（水平/垂直随机），输入/输出 [0,1]
    """
    if random.random() > prob:
        return img
    k = random.randint(kmin, kmax)
    if k % 2 == 0:
        k += 1
    # 随机选择水平或垂直核
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
    训练集增强流水线：
      p1–p99 裁剪 -> /255 -> resize -> (可选) CLAHE(prob) -> 几何/光照增强 -> z-score
    所有增强仅作用于图像通道，mask 只做几何变换（保持0/1）
    """
    def __init__(
        self,
        img_size: int,
        clip_percentiles=(1.0, 99.0),
        level: str = "light",
        # light 参数
        rot_deg: float = 7,
        trans_slide: float = 0.05,
        scale_range=(0.95, 1.05),
        brightness: float = 0.10,
        contrast: float = 0.10,
        gamma: float = 0.10,
        motion_blur_prob: float = 0.10,
        clahe_prob: float = 0.20,
        # strong 备用参数（当 level="strong" 时覆盖）
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
        # 1) 强度裁剪 -> [0,1]
        image = percentile_clip(image, self.clip_percentiles[0], self.clip_percentiles[1])
        image = scale_to_01(image)

        # 2) resize（先做几何一致的基准尺寸，增强在此尺寸上进行）
        image, mask = resize_image_and_mask(image, mask, self.img_size)

        # 3) CLAHE (prob)
        if random.random() < self.params["clahe_prob"]:
            image = apply_clahe(image, clip_limit=2.0, tile_grid_size=8)

        # 4) 几何增强（旋转/平移/缩放）
        image, mask = random_affine(
            image, mask,
            rot_deg=self.params["rot_deg"],
            trans_ratio=self.params["trans_slide"],
            scale_range=self.params["scale_range"]
        )

        # 5) 光照增强（brightness/contrast/gamma）
        image = random_brightness_contrast_gamma(
            image,
            brightness=self.params["brightness"],
            contrast=self.params["contrast"],
            gamma=self.params["gamma"]
        )

        # 6) 运动模糊（prob）
        image = random_motion_blur(image, prob=self.params["motion_blur_prob"], kmin=3, kmax=7)

        # 7) z-score（mask 不变）
        if self.znorm is not None:
            image = self.znorm(image)

        out = {"image": image.astype(np.float32), "mask": None if mask is None else mask.astype(np.float32),
               "label": int(label), "pid": pid}
        return out


class EvalTransform:
    """
    验证/测试集：确定性预处理（无随机增强）
      p1–p99 裁剪 -> /255 -> resize -> z-score
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
# 工厂函数：根据 cfg 生成 transform
# ------------------------------

def build_transforms(cfg, split: str, zscore_stats: Optional[dict] = None):
    """
    Args:
        cfg: 读取自 YAML 的配置对象/字典
        split: "train" / "val" / "test"
        zscore_stats: {"mean": float, "std": float}，训练后保存，用于 val/test
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

    # 包装成与 dataset.py 兼容的 callable
    def _wrapper(**data):
        return tfm(**data)

    return _wrapper
