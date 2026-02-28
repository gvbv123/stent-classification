import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from typing import Optional, Tuple

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _find_layer(self):
        layer = self.model
        for attr in self.target_layer.split("."):
            if attr.endswith("]"):
                base, idx = attr[:-1].split("[")
                layer = getattr(layer, base)[int(idx)]
            else:
                layer = getattr(layer, attr)
        return layer

    def _register_hooks(self):
        layer = self._find_layer()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        layer.register_forward_hook(forward_hook)
        layer.register_backward_hook(backward_hook)

    def generate(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)

        if logits.ndim == 2:
            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()
            score = logits[:, class_idx]
        else:
            score = logits.squeeze()

        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze().cpu().numpy()

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.size(3), x.size(2)))

        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

    def compute_roi_energy(self, cam: np.ndarray, mask: np.ndarray) -> float:
        if mask is None:
            return 0.0

        if cam.shape != mask.shape:
            mask = cv2.resize(mask, (cam.shape[1], cam.shape[0]), interpolation=cv2.INTER_NEAREST)

        total_energy = np.sum(cam)
        roi_energy = np.sum(cam * mask)

        energy_ratio = roi_energy / (total_energy + 1e-8)
        return float(energy_ratio)

def overlay_heatmap(img, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    if img.ndim == 3 and img.shape[0] == 2:
        img_disp = img[0]
    else:
        img_disp = img

    if img_disp.max() <= 1.0:
        img_uint8 = (img_disp * 255).astype(np.uint8)
    else:
        img_uint8 = img_disp.astype(np.uint8)

    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), colormap)
    overlay = cv2.addWeighted(img_uint8, 1 - alpha, heatmap, alpha, 0)
    return overlay
