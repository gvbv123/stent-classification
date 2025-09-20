import os
import cv2
import numpy as np

def make_gallery(cam_list, labels, preds, probs, pids, save_path, ncols=3):
    """
    把多张 CAM 图拼接成画廊
    Args:
        cam_list: List[np.ndarray], 每张(H,W,3) BGR
        labels: List[int], 真值
        preds: List[int], 预测
        probs: List[float], 预测概率
        pids: List[str], 病人ID
        save_path: 保存路径
    """
    n = len(cam_list)
    nrows = int(np.ceil(n / ncols))
    h, w, _ = cam_list[0].shape
    canvas = np.ones((nrows * h, ncols * w, 3), dtype=np.uint8) * 255

    for i, cam in enumerate(cam_list):
        r = i // ncols
        c = i % ncols
        y0, y1 = r * h, (r + 1) * h
        x0, x1 = c * w, (c + 1) * w

        canvas[y0:y1, x0:x1] = cam

        # 文字标签
        text = f"ID:{pids[i]} L={labels[i]} P={preds[i]} Pr={probs[i]:.2f}"
        cv2.putText(canvas, text, (x0 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, canvas)
    print(f"[CAM Gallery] Saved -> {save_path}")
