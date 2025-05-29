import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, exp_channels, out_channels, kernel_size, stride, se=True, nl='RE'):
        super(Bottleneck, self).__init__()
        padding = (kernel_size - 1) // 2
        self.use_res_connect = stride == 1 and in_channels == out_channels

        if nl == 'RE':
            nlin_layer = nn.ReLU
        elif nl == 'HS':
            nlin_layer = nn.Hardswish
        else:
            raise NotImplementedError

        layers = []
        # Expand
        if exp_channels != in_channels:
            layers.extend([
                nn.Conv2d(in_channels, exp_channels, 1, bias=False),
                nn.BatchNorm2d(exp_channels),
                nlin_layer(inplace=True)
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(exp_channels, exp_channels, kernel_size, stride, padding, groups=exp_channels, bias=False),
            nn.BatchNorm2d(exp_channels),
            nlin_layer(inplace=True)
        ])

        # SE
        if se:
            layers.append(SELayer(exp_channels))

        # Project
        layers.extend([
            nn.Conv2d(exp_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super(MobileNetV3Small, self).__init__()

        # First layer
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True)
        )

        # Bottleneck layers
        self.features.extend([
            Bottleneck(16, 16, 16, 3, 2, True, 'RE'),
            Bottleneck(16, 72, 24, 3, 2, False, 'RE'),
            Bottleneck(24, 88, 24, 3, 1, False, 'RE'),
            Bottleneck(24, 96, 40, 5, 2, True, 'HS'),
            Bottleneck(40, 240, 40, 5, 1, True, 'HS'),
            Bottleneck(40, 240, 40, 5, 1, True, 'HS'),
            Bottleneck(40, 120, 48, 5, 1, True, 'HS'),
            Bottleneck(48, 144, 48, 5, 1, True, 'HS'),
            Bottleneck(48, 288, 96, 5, 2, True, 'HS'),
            Bottleneck(96, 576, 96, 5, 1, True, 'HS'),
            Bottleneck(96, 576, 96, 5, 1, True, 'HS'),
        ])

        # Last layers
        self.conv = nn.Sequential(
            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            nn.Hardswish(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(576, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                # 添加bias存在性检查
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
