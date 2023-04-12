import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from torchvision import models


def freeze_model(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


class SqueezeNetBackbone(nn.Module):
    def __init__(self, freeze: bool = True):
        super().__init__()
        self.isfreeze = freeze
        self.net = self.load_model()
        if freeze:
            freeze_model(self.net)

    @staticmethod
    def load_model():
        net = models.squeezenet1_1(pretrained=True)
        del net.classifier
        return net

    def forward(self, x):
        return self.net.features(x)


class SqueezeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x


class ResNetBackbone(nn.Module):
    def __init__(self, freeze: bool = True):
        super().__ini__()
        self.net = self.load_model()
        self.isfreeze = freeze
        if freeze:
            freeze_model(self.net)

    @staticmethod
    def load_model():
        net = models.resnet50(pretrained=True)
        net.fc = nn.Identity()
        return net

    def forward(self, x):
        x = self.net(x)
        return x


class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name: str = "efficientnet-b4", freeze: bool = True):
        super().__init__()
        self.version = model_name
        self.isfreeze = freeze
        self.net = self.load_model(model_name)
        if freeze:
            freeze_model_(self.net)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def load_model(model_name: str):
        net = EfficientNet.from_pretrained(model_name, num_classes=1)
        del net._fc
        return net

    def forward(self, x):
        bs = x.size(0)
        x = self.net.extract_features(x)
        x = self.avg_pooling(x)
        x = x.view(bs, -1)
        return x
