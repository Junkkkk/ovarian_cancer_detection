import torch
import torch.nn as nn
import torch.nn.init as init
from .backbone import (
    EfficientNetBackbone,
    SqueezeNetBackbone,
    SqueezeNetClassifier,
    ResNetBackbone,
)

BACKBONE_EMBEDDING_DIM = {
    "efficientnet-b5": 2048,
    "efficientnet-b4": 1792,
    "efficientnet-b3": 1536,
    "efficientnet-b2": 1408,
    "efficientnet-b1": 1280,
    "efficientnet-b0": 1280,
    "squeezenet": 512,
    "resnet": 2048,
}


class CNN(nn.Module):
    def __init__(self, backbone: str = "squeezenet", freeze: bool = False):
        super().__init__()
        self.backbone = backbone
        self.backbone_dim = BACKBONE_EMBEDDING_DIM[backbone]
        self.isfreeze = freeze
        self.net = self.init_model()

    def forward(self, x):
        return self.net(x)

    def init_model(self):
        if "efficientnet" in self.backbone:
            encoder = EfficientNetBackbone(model_name=self.backbone, freeze=self.isfreeze)
            classifier = nn.Linear(self.backbone_dim, 2)
        elif self.backbone == "squeezenet":
            encoder = SqueezeNetBackbone(freeze=self.isfreeze)
            classifier = SqueezeNetClassifier(num_classes=2)
        else:
            raise ValueError(f"Invaild backbone name:", {self.backbone})

        model = nn.Sequential(encoder, classifier)
        return model
