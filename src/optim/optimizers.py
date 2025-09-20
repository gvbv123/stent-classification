import torch
import torch.optim as optim

def get_optimizer(cfg, model_params):
    """
    根据配置创建优化器
    cfg["OPTIM"]["OPTIMIZER"]: "adamw" / "sgd" / "adam"
    cfg["OPTIM"]["LR"]: 初始学习率
    cfg["OPTIM"]["WEIGHT_DECAY"]: 权重衰减
    """
    opt_name = cfg["OPTIM"]["OPTIMIZER"].lower()
    lr = float(cfg["OPTIM"]["LR"])
    wd = float(cfg["OPTIM"]["WEIGHT_DECAY"])

    if opt_name == "adamw":
        return optim.AdamW(model_params, lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        return optim.Adam(model_params, lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return optim.SGD(model_params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    else:
        raise ValueError(f"未知优化器: {opt_name}")
