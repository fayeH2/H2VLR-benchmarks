"""Backbone feature extractor for PatchCore (torchvision Wide ResNet-50)."""

import torch
import torch.nn as nn
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

_SEQUENCE = (4, 8, 16)  # spatial strides of layer1, layer2, layer3 at 224px input


class FeatureExtractor(nn.Module):
    """wide_resnet50_2 patch features; concatenates layer2+layer3 by default.

    Forward returns [B, G, C] where G is the number of spatial patches of the
    deepest selected layer and C the concatenated channel count.
    """

    def __init__(self, layer="layer23", pretrained=True):
        super().__init__()
        backbone = wide_resnet50_2(
            weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        pool = {
            "layer12": [backbone.layer1, backbone.layer2],
            "layer23": [backbone.layer2, backbone.layer3],
            "layer123": [backbone.layer1, backbone.layer2, backbone.layer3],
        }[layer]
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                  backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = (backbone.layer1, backbone.layer2,
                                                 backbone.layer3)
        self.selected = pool
        channels = {
            "layer12": [256, 512],
            "layer23": [512, 1024],
            "layer123": [256, 512, 1024],
        }[layer]
        self.out_dim = sum(channels)
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    @torch.no_grad()
    def collect(self, images):
        """Return a list with each selected layer's (B, C, H, W) tensor."""
        features = self.stem(images)
        layers = [self.layer1, self.layer2, self.layer3]
        selected = []
        for layer in layers:
            features = layer(features)
            if layer in self.selected:
                selected.append(features)
        return selected

    @torch.no_grad()
    def forward(self, images):
        selected = self.collect(images)
        grid = selected[-1].shape[-2:]
        # align all selected layers to the deepest grid via adaptive average pooling
        aligned = [torch.nn.functional.adaptive_avg_pool2d(feat, grid) if feat.shape[-2:] != grid
                   else feat for feat in selected]
        # [B, G, C] with G grid of deepest selected layer
        return torch.cat([feat.flatten(2).transpose(1, 2) for feat in aligned], dim=-1)