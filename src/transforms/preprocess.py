import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any

# ------------------------------
# 基础预处理函数
# ------------------------------

def percentile_clip(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """
    按百分位裁剪强度（用于8-bit灰度）。不改变 dtype，仅裁剪。
    """
    assert img.ndim == 2, "expect HxW grayscale"
    lo = np.percentile(img, p_low)
    hi = np.percentile(img, p_high)
    if hi <= lo:
        return img.copy()
    img_clipped = np.clip(img, lo, hi)
    # 线性拉伸到 [0, 255]
    img_clipped = (img_clipped - lo) / (hi - lo + 1e-8)
    img_clipped = (img_clipped * 255.0).astype(np.float32)
    return img_clipped


def scale_to_01(img: np.ndarray) -> np.ndarray:
    """
    将 0-255 浮点图缩放到 [0,1]
    """
    img = img.astype(np.float32)
    return img / 255.0


def resize_image_and_mask(
    img: np.ndarray,
    mask: Optional[np.ndarray],
    size: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    统一尺寸到 (size, size)
    图像：双线性；掩膜：最近邻（保持二值）
    """
    img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    mask_resized = None
    if mask is not None:
        mask_resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img_resized, mask_resized


class DatasetZScore:
    """
    用训练集统计到的 mean/std 对图像通道 z-score 标准化（掩膜不变）
    """
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std) if std is not None and std > 0 else 1.0

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # img: [0,1] float
        return (img - self.mean) / (self.std + 1e-8)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """
    对 [0,1] 浮点图应用 CLAHE（转换到 0-255 再返回 [0,1]）
    """
    img_255 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    img_eq = clahe.apply(img_255)
    return img_eq.astype(np.float32) / 255.0


# ------------------------------
# 组合式 Transform（不依赖 Albumentations）
# 用于 val/test 或作为增强前的公共预处理
# ------------------------------

class BasePreprocess:
    """
    公共预处理流水线：
      p1–p99 裁剪 -> /255 -> resize -> (可选) z-score
    """
    def __init__(
        self,
        img_size: int,
        clip_percentiles: Tuple[float, float] = (1.0, 99.0),
        use_zscore: bool = True,
        zscore_stats: Optional[Dict[str, float]] = None,
    ):
        self.img_size = int(img_size)
        self.clip_percentiles = clip_percentiles
        self.use_zscore = use_zscore
        self.znorm = None
        if use_zscore and zscore_stats is not None:
            self.znorm = DatasetZScore(zscore_stats["mean"], zscore_stats["std"])

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray], label: int, pid: str) -> Dict[str, Any]:
        # 强度裁剪
        image = percentile_clip(image, self.clip_percentiles[0], self.clip_percentiles[1])
        # [0,1]
        image = scale_to_01(image)
        # resize
        image, mask = resize_image_and_mask(image, mask, self.img_size)
        # z-score
        if self.use_zscore and self.znorm is not None:
            image = self.znorm(image)
        # 返回
        out = {"image": image.astype(np.float32), "mask": None if mask is None else mask.astype(np.float32),
               "label": int(label), "pid": pid}
        return out
