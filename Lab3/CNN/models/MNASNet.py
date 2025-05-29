import math

import torch.nn as nn


class _SepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(_SepConv, self).__init__()
        self.conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                      padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class _SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(_SELayer, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class _MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, use_se=True):
        super(_MBConv, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        exp_channels = int(in_channels * expand_ratio)
        self.conv = nn.Sequential(
            # Expansion
            nn.Conv2d(in_channels, exp_channels, 1, bias=False),
            nn.BatchNorm2d(exp_channels),
            nn.ReLU6(inplace=True),
            # Depthwise
            nn.Conv2d(exp_channels, exp_channels, kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2, groups=exp_channels, bias=False),
            nn.BatchNorm2d(exp_channels),
            nn.ReLU6(inplace=True),
            # SE
            _SELayer(exp_channels) if use_se else nn.Identity(),
            # Project
            nn.Conv2d(exp_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MNASNet(nn.Module):
    def __init__(self, alpha=0.5, num_classes=10, dropout_rate=0.2):
        super(MNASNet, self).__init__()

        def depth(d):
            return max(int(d * alpha), 1)

        # First layer
        self.layers = nn.Sequential(
            nn.Conv2d(1, depth(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(depth(32)),
            nn.ReLU6(inplace=True),
            _SepConv(depth(32), depth(16), 3, padding=1),
        )

        # MBConv blocks
        mb_config = [
            # k, t, c, n, s, se
            [3, 3, 24, 3, 2, False],  # MBConv3 3x24
            [5, 3, 40, 3, 2, True],  # MBConv5 3x40
            [5, 6, 80, 3, 2, False],  # MBConv5 6x80
            [3, 6, 96, 2, 1, True],  # MBConv3 6x96
            [5, 6, 192, 4, 2, True],  # MBConv5 6x192
            [5, 6, 320, 1, 1, True],  # MBConv5 6x320
        ]

        in_channels = depth(16)
        for k, t, c, n, s, se in mb_config:
            out_channels = depth(c)
            for i in range(n):
                stride = s if i == 0 else 1
                self.layers.add_module(f'mb_{k}_{t}_{c}_{i}',
                                       _MBConv(in_channels, out_channels, k, stride, t, se))
                in_channels = out_channels

        # Last layer
        last_channels = depth(1280)
        self.layers.add_module('conv_last', nn.Sequential(
            nn.Conv2d(in_channels, last_channels, 1, bias=False),
            nn.BatchNorm2d(last_channels),
            nn.ReLU6(inplace=True)
        ))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(last_channels, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
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
