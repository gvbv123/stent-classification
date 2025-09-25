from .backbones import get_backbone
from .heads import ClassificationHead

class ModelFactory:
    @staticmethod
    def build(cfg):
        """
        Build a complete model based on the configuration.
        cfg: dict
          - MODEL.BACKBONE
          - INPUT.CHANNELS
          - MODEL.DROPOUT
        """
        backbone_name = cfg["MODEL"]["BACKBONE"]
        in_ch = int(cfg["INPUT"]["CHANNELS"])
        pretrained = bool(cfg["MODEL"]["PRETRAINED"])
        dropout = float(cfg["MODEL"]["DROPOUT"])

        backbone, feat_dim = get_backbone(backbone_name, in_channels=in_ch, pretrained=pretrained)
        head = ClassificationHead(feat_dim, num_classes=2, dropout=dropout)

        import torch.nn as nn
        model = nn.Sequential(
            backbone,
            head
        )
        return model
