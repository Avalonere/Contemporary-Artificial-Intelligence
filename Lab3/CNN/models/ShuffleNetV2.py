import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    batch_size, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        branch_features = out_channels // 2

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=self.stride,
                          padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels if stride > 1 else branch_features, branch_features,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride,
                      padding=1, groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, width_multiplier=0.5, num_classes=10, dropout_rate=0.2):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [-1, 24,
                                   int(116 * width_multiplier),  # stage 2
                                   int(232 * width_multiplier),  # stage 3
                                   int(464 * width_multiplier),  # stage 4
                                   int(1024 * width_multiplier)]  # conv5

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.stage_out_channels[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[1]),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 构建网络层
        self.stage2 = self._make_stage(2)
        self.stage3 = self._make_stage(3)
        self.stage4 = self._make_stage(4)

        self.conv5 = nn.Sequential(
            nn.Conv2d(self.stage_out_channels[-2], self.stage_out_channels[-1],
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1]),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.stage_out_channels[-1], num_classes)

    def _make_stage(self, stage):
        layers = []
        in_channels = self.stage_out_channels[stage - 1]
        out_channels = self.stage_out_channels[stage]

        for i in range(self.stage_repeats[stage - 2]):
            stride = 2 if i == 0 else 1
            layers.append(InvertedResidual(in_channels, out_channels, stride))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)

        x = x.mean([2, 3])  # global average pooling
        x = self.dropout(x)
        x = self.fc(x)
        return x
