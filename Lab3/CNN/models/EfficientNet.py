# EfficientNet.py
import math

import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio=0.25, drop_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.drop_rate = drop_rate
        self.use_residual = stride == 1 and in_channels == out_channels

        exp_channels = int(in_channels * expand_ratio)

        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, exp_channels, 1, bias=False),
            nn.BatchNorm2d(exp_channels),
            nn.SiLU(inplace=True)
        ) if expand_ratio != 1 else nn.Identity()

        self.dwconv = nn.Sequential(
            nn.Conv2d(exp_channels, exp_channels, kernel_size, stride,
                      kernel_size // 2, groups=exp_channels, bias=False),
            nn.BatchNorm2d(exp_channels),
            nn.SiLU(inplace=True)
        )

        self.se = SEModule(exp_channels, int(1 / se_ratio))

        self.project_conv = nn.Sequential(
            nn.Conv2d(exp_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        identity = x

        x = self.expand_conv(x)
        x = self.dwconv(x)
        x = self.se(x)
        x = self.project_conv(x)

        if self.use_residual:
            if self.training and self.drop_rate > 0:
                x = self.drop(x)
            x = x + identity
        return x


class EfficientNet(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super(EfficientNet, self).__init__()

        # B0配置参数
        self.config = [
            # k, c, n, s, e
            [3, 16, 1, 1, 1],  # MBConv1
            [3, 24, 2, 2, 6],  # MBConv6
            [5, 40, 2, 2, 6],  # MBConv6
            [3, 80, 3, 2, 6],  # MBConv6
            [5, 112, 3, 1, 6],  # MBConv6
            [5, 192, 4, 2, 6],  # MBConv6
            [3, 320, 1, 1, 6]  # MBConv6
        ]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )

        # Blocks
        layers = []
        in_channels = 32

        for k, c, n, s, e in self.config:
            out_channels = c
            layers.extend([
                MBConvBlock(in_channels if i == 0 else out_channels,
                            out_channels, k, s if i == 0 else 1, e,
                            drop_rate=dropout_rate)
                for i in range(n)
            ])
            in_channels = out_channels

        self.blocks = nn.Sequential(*layers)

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(1280, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
