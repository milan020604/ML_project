import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    A small ResNet-style residual block.

    Each block has:
    - two 3x3 convolutional layers
    - batch normalization after each convolution
    - ReLU activation
    - a shortcut connection that adds the input back to the output

    If the number of channels changes, or if the spatial size is reduced,
    the shortcut uses a 1x1 convolution so the shapes match.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class CNN(nn.Module):
    """
    Small ResNet-style CNN for CIFAR-10.

    Input shape:
        (batch_size, 3, 32, 32)

    Output shape:
        (batch_size, 10)

    Architecture:
    - initial 3x3 convolutional stem
    - 3 residual blocks
    - global average pooling
    - final linear classifier
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                30,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(30),
            nn.ReLU(inplace=True),
        )

        self.block1 = ResidualBlock(30, 30, stride=1)
        self.block2 = ResidualBlock(30, 60, stride=2)
        self.block3 = ResidualBlock(60, 120, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def get_model():
    return CNN()