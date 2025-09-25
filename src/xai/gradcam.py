import torch
import torch.nn as nn
import numpy as np
import cv2
import os

class GradCAM:
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _find_layer(self):
        """Find the target layer by its string path."""
        layer = self.model
        for attr in self.target_layer.split("."):
            if attr.endswith("]"):  # e.g., layer4[-1].conv2
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
        """
        Generate a heatmap for the class index (default is the argmax of logits).
        Args:
            x: (1,C,H,W) input tensor
            class_idx: Target class index (default is the logits argmax)
        Returns:
            heatmap: (H,W) heatmap scaled to [0, 1]
        """
        self.model.zero_grad()
        logits = self.model(x)  # (1,2) or (1,)
        if logits.ndim == 2:
            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()
            score = logits[:, class_idx]
        else:
            score = logits.squeeze()

        score.backward(retain_graph=True)

        # GAP on gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (C,1,1)
        cam = (weights * self.activations).sum(dim=1).squeeze().cpu().numpy()

        cam = np.maximum(cam, 0)  # ReLU
        cam = cv2.resize(cam, (x.size(3), x.size(2)))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

def overlay_heatmap(img, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay the CAM heatmap onto the original image.
    Args:
        img: (H,W) float or uint8, range [0,1] or [0,255]
        cam: (H,W) float, range [0,1]
    """
    if img.max() <= 1.0:
        img_disp = (img * 255).astype(np.uint8)
    else:
        img_disp = img.astype(np.uint8)
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), colormap)
    overlay = cv2.addWeighted(img_disp, 1 - alpha, heatmap, alpha, 0)
    return overlay
