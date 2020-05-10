import math

import megengine.functional as F
import megengine.module as M

from layers.batch_norm import FrozenBatchNorm2d

has_bias = False

class Bottleneck(M.Module):
    def __init__(
            self, in_channels, bottleneck_channels, out_channels,
            stride, dilation=1):
        super(Bottleneck, self).__init__()

        self.downsample = None
        self.downsample = (
            M.Identity()
            if in_channels == out_channels and stride == 1
            else M.Sequential(
                M.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=has_bias),
                FrozenBatchNorm2d(out_channels),
            )
        )

        self.conv1 = M.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=has_bias)
        self.bn1 = FrozenBatchNorm2d(bottleneck_channels)
        self.conv2 = M.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride,
                padding=dilation, bias=has_bias, dilation=dilation)
        self.bn2 = FrozenBatchNorm2d(bottleneck_channels)
        self.conv3 = M.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=has_bias)
        self.bn3 = FrozenBatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        identity = self.downsample(identity)

        x += identity
        x = F.relu(x)

        return x

class ResNet50(M.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = M.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=has_bias)
        self.bn1 = FrozenBatchNorm2d(64)
        self.maxpool = M.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block_counts = [3, 4, 6, 3]
        bottleneck_channels_list = [64, 128, 256, 512]
        out_channels_list = [256, 512, 1024, 2048]
        stride_list = [1, 2, 2, 2]
        in_channels = 64
        self.layer1 = self._make_layer(block_counts[0], 64,
            bottleneck_channels_list[0], out_channels_list[0], stride_list[0])
        self.layer2 = self._make_layer(block_counts[1], out_channels_list[0],
            bottleneck_channels_list[1], out_channels_list[1], stride_list[1])
        self.layer3 = self._make_layer(block_counts[2], out_channels_list[1],
            bottleneck_channels_list[2], out_channels_list[2], stride_list[2])
        self.layer4 = self._make_layer(block_counts[3], out_channels_list[2],
            bottleneck_channels_list[3], out_channels_list[3], stride_list[3])
        
        for l in self.modules():
            if isinstance(l, M.Conv2d):
                M.init.msra_normal_(l.weight, mode="fan_in")
                if has_bias:
                    M.init.fill_(l.bias, 0)

    def _make_layer(self, num_blocks, in_channels, bottleneck_channels, out_channels, stride):
        layers = []
        for _ in range(num_blocks):
            layers.append(Bottleneck(in_channels, bottleneck_channels, out_channels, stride))
            stride = 1
            in_channels = out_channels
        return M.Sequential(*layers)

    def forward(self, x):
        outputs = []
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        # blocks
        x = self.layer1(x)
        outputs.append(x)
        x = self.layer2(x)
        outputs.append(x)
        x = self.layer3(x)
        outputs.append(x)
        x = self.layer4(x)
        outputs.append(x)
        return outputs

