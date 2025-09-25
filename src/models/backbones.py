import torch
import torch.nn as nn
import torchvision.models as tvm
import numpy as np

try:
    import timm
    _has_timm = True
except ImportError:
    _has_timm = False


def _replace_module(parent, name, new):
    setattr(parent, name, new)

def adapt_first_conv(model: nn.Module, in_channels: int):
    if in_channels is None:
        return model

    # Traverse and find the first Conv2d layer, its parent module, and its name in the parent module
    first_conv, parent, name = None, None, None
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Find the name of the module in the parent and the parent reference
            parts = module_name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            name = parts[-1]
            first_conv = module
            break

    if first_conv is None:
        raise ValueError("Unable to find the first convolution layer, please adapt manually")

    old = first_conv
    if old.in_channels == in_channels:
        return model  

    new_conv = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups if old.groups == 1 else 1,  
        bias=(old.bias is not None)
    )

    with torch.no_grad():
        W = old.weight.data  # [out_c, in_c_old, k, k]
        if in_channels == 1 and W.shape[1] >= 3:
            # Grayscale: average RGB weights along the channel dimension
            new_conv.weight.copy_(W.mean(dim=1, keepdim=True))
        elif in_channels > W.shape[1]:
            # More channels: repeat and trim, scale by the repetition factor
            rep = int(np.ceil(in_channels / W.shape[1]))
            W_rep = W.repeat(1, rep, 1, 1)[:, :in_channels, :, :] / rep
            new_conv.weight.copy_(W_rep)
        else:
            # Fewer channels: directly take the first in_channels
            new_conv.weight.copy_(W[:, :in_channels, :, :])

        if old.bias is not None:
            new_conv.bias.copy_(old.bias.data)

    # Replace the first convolution with the new one
    _replace_module(parent, name, new_conv)
    return model


def get_backbone(name: str, in_channels: int = 1, pretrained: bool = True):
    """
    Return the backbone and the output feature dimension.
    """
    name = name.lower()
    if name == "resnet18":
        model = tvm.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        model = adapt_first_conv(model, in_channels)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim

    elif name == "efficientnet_b0":
        if not _has_timm:
            raise ImportError("timm is required: pip install timm")
        model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0, global_pool="avg")
        model = adapt_first_conv(model, in_channels)
        feat_dim = model.num_features
        return model, feat_dim

    elif name == "efficientnet_b2":
        if not _has_timm:
            raise ImportError("timm is required: pip install timm")
        model = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        model = adapt_first_conv(model, in_channels)  # Adapt for 1/2 channels
        feat_dim = model.num_features
        return model, feat_dim

    elif name == "efficientnet_b3":
        if not _has_timm:
            raise ImportError("timm is required: pip install timm")
        model = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        model = adapt_first_conv(model, in_channels)  # Adapt for 1/2 channels
        feat_dim = model.num_features
        return model, feat_dim

    elif name == "convnext_tiny":
        if not _has_timm:
            raise ImportError("timm is required: pip install timm")
        model = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=0, global_pool="avg")
        model = adapt_first_conv(model, in_channels)
        feat_dim = model.num_features
        return model, feat_dim

    elif name == "densenet121":
        model = tvm.densenet121(weights="IMAGENET1K_V1" if pretrained else None)
        model.features[-1] = nn.Identity()  
        model.classifier = nn.Identity()
        model = adapt_first_conv(model, in_channels)
        feat_dim = 1024 
        return model, feat_dim

    elif name == "swin_tiny":
        if not _has_timm:
            raise ImportError("timm is required: pip install timm")
        model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=in_channels, 
        )
        feat_dim = model.num_features
        return model, feat_dim

    else:
        raise ValueError(f"Unknown backbone: {name}")
