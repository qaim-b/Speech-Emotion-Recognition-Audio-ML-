from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D_SER(nn.Module):
"""A compact 2D-CNN over (freq x time) with global pooling."""
def __init__(self, n_classes: int, in_channels: int = 1):
super().__init__()
self.feat = nn.Sequential(
nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
)
self.fc = nn.Linear(128, n_classes)


def forward(self, x):
# x: (B, C=1, F, T)
h = self.feat(x)
h = h.view(h.size(0), -1)
return self.fc(h)
