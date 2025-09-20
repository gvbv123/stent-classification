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
    """把 parent.name 指向的子模块替换成 new"""
    setattr(parent, name, new)

def adapt_first_conv(model: nn.Module, in_channels: int):
    """
    通用适配：把骨干网络的第一层 Conv2d 改成 in_channels
    - 若原来是3通道，且现在是1通道：对权重在通道维求均值
    - 若现在通道数更大：重复/截取并做尺度归一
    """
    if in_channels is None:
        return model

    # 遍历找到第一个 Conv2d 以及它的父模块和在父模块里的名字
    first_conv, parent, name = None, None, None
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 找到这个模块在父模块中的名字与父引用
            parts = module_name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            name = parts[-1]
            first_conv = module
            break

    if first_conv is None:
        raise ValueError("无法找到第一层卷积，请手工适配")

    old = first_conv
    if old.in_channels == in_channels:
        return model  # 已一致

    new_conv = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups if old.groups == 1 else 1,  # 保守处理分组
        bias=(old.bias is not None)
    )

    with torch.no_grad():
        W = old.weight.data  # [out_c, in_c_old, k, k]
        if in_channels == 1 and W.shape[1] >= 3:
            # 灰度：把RGB权重在通道维平均
            new_conv.weight.copy_(W.mean(dim=1, keepdim=True))
        elif in_channels > W.shape[1]:
            # 通道更多：重复并截断，同时按重复次数归一
            rep = int(np.ceil(in_channels / W.shape[1]))
            W_rep = W.repeat(1, rep, 1, 1)[:, :in_channels, :, :] / rep
            new_conv.weight.copy_(W_rep)
        else:
            # 通道更少：直接取前 in_channels 个
            new_conv.weight.copy_(W[:, :in_channels, :, :])

        if old.bias is not None:
            new_conv.bias.copy_(old.bias.data)

    # 用新的卷积替换掉原来的第一层
    _replace_module(parent, name, new_conv)
    return model


def get_backbone(name: str, in_channels: int = 1, pretrained: bool = True):
    """
    返回 backbone + 输出特征维度
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
            raise ImportError("需要安装 timm: pip install timm")
        model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0, global_pool="avg")
        model = adapt_first_conv(model, in_channels)
        feat_dim = model.num_features
        return model, feat_dim

    elif name == "efficientnet_b2":
        if not _has_timm:
            raise ImportError("需要安装 timm: pip install timm")
        model = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        model = adapt_first_conv(model, in_channels)  # 适配 1/2 通道
        feat_dim = model.num_features
        return model, feat_dim

    elif name == "efficientnet_b3":
        if not _has_timm:
            raise ImportError("需要安装 timm: pip install timm")
        model = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        model = adapt_first_conv(model, in_channels)  # 适配 1/2 通道
        feat_dim = model.num_features
        return model, feat_dim


    elif name == "convnext_tiny":
        if not _has_timm:
            raise ImportError("需要安装 timm: pip install timm")
        model = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=0, global_pool="avg")
        model = adapt_first_conv(model, in_channels)
        feat_dim = model.num_features
        return model, feat_dim

    elif name == "densenet121":
        model = tvm.densenet121(weights="IMAGENET1K_V1" if pretrained else None)
        model.features[-1] = nn.Identity()  # 去掉最后ReLU
        model.classifier = nn.Identity()
        model = adapt_first_conv(model, in_channels)
        feat_dim = 1024  # densenet121 特征维度
        return model, feat_dim

    elif name == "swin_tiny":
        if not _has_timm:
            raise ImportError("需要安装 timm: pip install timm")
        model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=in_channels,  # ← 加上这一句以适配 1/2 通道
        )
        feat_dim = model.num_features
        return model, feat_dim

    else:
        raise ValueError(f"未知 backbone: {name}")
