"""The two heterogeneous CNN architectures from the supplied legacy script."""

import torch.nn as nn


class SmallCIFARCNN_A(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.fc(self.feat(x).flatten(1))


class SmallCIFARCNN_B(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.05),
            nn.Conv2d(96, 160, 3, padding=1), nn.BatchNorm2d(160), nn.ReLU(),
            nn.Conv2d(160, 160, 3, padding=1), nn.BatchNorm2d(160), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.05),
            nn.Conv2d(160, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.fc(self.feat(x).flatten(1))


def make_clients(count, classes):
    return [(SmallCIFARCNN_A if k % 2 == 0 else SmallCIFARCNN_B)(classes)
            for k in range(count)]
