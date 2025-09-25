from .bce_posweight import BCEWithPosWeight
from .focal import FocalLoss

def get_loss(cfg, pos_weight=None):
    """
    Build the loss function based on the configuration.
    cfg["LOSS"]["TYPE"]: "bce_posweight" / "focal"
    cfg["LOSS"]["FOCAL_GAMMA"]
    cfg["LOSS"]["FOCAL_ALPHA"]
    """
    loss_type = cfg["LOSS"]["TYPE"].lower()

    if loss_type == "bce_posweight":
        return BCEWithPosWeight(pos_weight=pos_weight)

    elif loss_type == "focal":
        gamma = cfg["LOSS"].get("FOCAL_GAMMA", 2.0)
        alpha = cfg["LOSS"].get("FOCAL_ALPHA", None)
        return FocalLoss(gamma=gamma, alpha=alpha)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
