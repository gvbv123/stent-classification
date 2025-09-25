import torch
import torch.optim as optim

def get_optimizer(cfg, model_params):
    """
    Create an optimizer based on the configuration.
    cfg["OPTIM"]["OPTIMIZER"]: "adamw" / "sgd" / "adam"
    cfg["OPTIM"]["LR"]: Initial learning rate
    cfg["OPTIM"]["WEIGHT_DECAY"]: Weight decay
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
        raise ValueError(f"Unknown optimizer: {opt_name}")
