import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def model_factory(args, num_classes, in_shape):
    if args.activation == "relu":
        act = torch.nn.ReLU()
    elif args.activation == "leaky":
        act = torch.nn.LeakyReLU()
    else:
        raise NotImplementedError(args.activation)

    if args.model == "vgg11":
        return VGG11(args.ic_index, act, num_classes, args.batch_norm, in_shape[0])
    elif args.model == "vgg16":
        return VGG16(args.ic_index, act, num_classes, args.batch_norm, in_shape[0])
    elif args.model == "vgg19":
        return VGG19(args.ic_index, act, num_classes, args.batch_norm, in_shape[0])
    elif args.model == "resnet18":
        return ResNet18(args.ic_index, act, num_classes, args.batch_norm, in_shape[0])
    elif args.model == "resnet34":
        return ResNet34(args.ic_index, act, num_classes, args.batch_norm, in_shape[0])
    elif args.model == "resnet50":
        return ResNet50(args.ic_index, act, num_classes, args.batch_norm, in_shape[0])
    else:
        raise NotImplementedError(args.model)


def flops_of_conv2d(conv, input_shape):
    out_h = (input_shape[1] + conv.padding[0]*2 -
          (conv.kernel_size[0]-1)) // conv.stride[0]
    out_w = (input_shape[2] + conv.padding[1]*2 -
          (conv.kernel_size[1]-1)) // conv.stride[1]
    out_shape = (conv.out_channels, out_h, out_w)
    return out_h * out_w * conv.weight.size().numel(), out_shape


class DenseBlock(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 bn=True,
                 act=nn.ReLU()):

        super().__init__()
        self.main = nn.Linear(in_features, out_features, bias)
        self.bn = nn.BatchNorm1d(out_features) if bn else None
        self.act = act

    def forward(self, x):
        x = self.main(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.act(x)
        return x

    def flops(self, input_shape=None):
        return self.main.in_features * self.main.out_features, input_shape


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 bn=True,
                 act=nn.ReLU()):

        super().__init__()
        self.main = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                padding,
                                bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.act = act

    def forward(self, x):
        x = self.main(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.act(x)
        return x

    def flops(self, input_shape):
        return flops_of_conv2d(self.main, input_shape)


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 bias=True,
                 bn=True,
                 act=nn.ReLU()):

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels) if bn else None
        self.act1 = act

        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels) if bn else None
        self.act2 = act

        self.conv_skip = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=1,
                                   stride=stride,
                                   padding=0,
                                   bias=True)

    def forward(self, x):
        skip = self.conv_skip(x)

        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = x + skip
        x = self.act2(x)
        return x

    def flops(self, input_shape):
        mid_flops, mid_shape = flops_of_conv2d(self.conv1, input_shape)
        out_flops, out_shape = flops_of_conv2d(self.conv2, mid_shape)
        skip_flops, _ = flops_of_conv2d(self.conv_skip, input_shape)
        return mid_flops + out_flops + skip_flops, out_shape


class ResBlock2(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride=1,
                 act=nn.ReLU()):

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               mid_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act1 = act

        self.convM = nn.Conv2d(mid_channels,
                               mid_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=True)
        self.bnM = nn.BatchNorm2d(mid_channels)
        self.actM = act

        self.conv2 = nn.Conv2d(mid_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = act

        self.conv_skip = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=1,
                                   stride=stride,
                                   padding=0,
                                   bias=True)

    def forward(self, x):
        skip = self.conv_skip(x)

        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.act1(x)

        x = self.convM(x)
        if self.bn2 is not None:
            x = self.bnM(x)
        x = self.actM(x)

        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)

        x = x + skip
        x = self.act2(x)
        return x

    def flops(self, input_shape):
        mid1_flops, mid1_shape = flops_of_conv2d(self.conv1, input_shape)
        mid2_flops, mid2_shape = flops_of_conv2d(self.convM, mid1_shape)
        out_flops, out_shape = flops_of_conv2d(self.conv2, mid2_shape)
        skip_flops, _ = flops_of_conv2d(self.conv_skip, input_shape)
        return mid1_flops + mid2_flops + out_flops + skip_flops, out_shape


class SDN(nn.Module):
    def __init__(self, num_classes, feature_layers, classifier_indices):
        super().__init__()

        self.feature = nn.ModuleList(feature_layers)
        classifiers = {}
        for index in classifier_indices:
            if isinstance(self.feature[index], ConvBlock):
                channels = self.feature[index].main.out_channels
            elif isinstance(self.feature[index], ResBlock):
                channels = self.feature[index].conv2.out_channels
            elif isinstance(self.feature[index], ResBlock2):
                channels = self.feature[index].conv2.out_channels
            classifiers[str(index)] = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                DenseBlock(channels, num_classes, bias=True, bn=False, act=nn.Identity())
            )
        self.classifiers = nn.ModuleDict(classifiers)

    def forward(self, x):
        y = []
        for layer in self.feature:
            x = layer(x)
            y.append(x)

        out = {}
        for index, layer in self.classifiers.items():
            out[int(index)] = layer(y[int(index)])

        return out


class VGG11(SDN):
    def __init__(self, ic_index, act, num_classes, batch_norm, input_channel):
        """
        ic_index: [0, ..., 7]
        """
        bias = False if batch_norm else True

        feature_layers = [
            # 32x32
            ConvBlock(input_channel, 64, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            # 16x16
            ConvBlock(64, 128, 3, stride=2, padding=1, bias=bias, bn=batch_norm, act=act),
            # 8x8
            ConvBlock(128, 256, 3, stride=2, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(256, 256, 3 ,stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            # 4x4
            ConvBlock(256, 512, 3, stride=2, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(512, 512, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            # 2x2
            ConvBlock(512, 512, 3, stride=2, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(512, 512, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act)
        ]
        super().__init__(num_classes, feature_layers, classifier_indices=[ic_index, -1])


class VGG16(SDN):
    def __init__(self, ic_index, act, num_classes, batch_norm, input_channel):
        """
        ic_index: [0, ..., 12]
        """
        bias = False if batch_norm else True

        feature_layers = [
            # 32x32
            ConvBlock(input_channel, 64, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(64, 64, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            # 16x16
            ConvBlock(64, 128, 3, stride=2, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(128, 128, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            # 8x8
            ConvBlock(128, 256, 3, stride=2, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(256, 256, 3 ,stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(256, 256, 3 ,stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            # 4x4
            ConvBlock(256, 512, 3, stride=2, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(512, 512, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(512, 512, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            # 2x2
            ConvBlock(512, 512, 3, stride=2, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(512, 512, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(512, 512, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act)
        ]
        super().__init__(num_classes, feature_layers, classifier_indices=[ic_index, -1])


class VGG19(SDN):
    def __init__(self, ic_index, act, num_classes, batch_norm, input_channel):
        """
        ic_index: [0, ..., 15]
        """
        bias = False if batch_norm else True
        feature_layers = [
            # 32x32 with cifar dataset
            ConvBlock(input_channel, 64, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(64, 64, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            # 16x16
            ConvBlock(64, 128, 3, stride=2, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(128, 128, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            # 8x8
            ConvBlock(128, 256, 3, stride=2, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(256, 256, 3 ,stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(256, 256, 3 ,stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(256, 256, 3 ,stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            # 4x4
            ConvBlock(256, 512, 3, stride=2, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(512, 512, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(512, 512, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(512, 512, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            # 2x2
            ConvBlock(512, 512, 3, stride=2, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(512, 512, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(512, 512, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            ConvBlock(512, 512, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act)
        ]
        super().__init__(num_classes, feature_layers, classifier_indices=[ic_index, -1])


class ResNet18(SDN):
    def __init__(self, ic_index, act, num_classes, batch_norm, input_channel):
        """
        ic_index: [0, 1, 2, 3, 4, 5, ..., 7]
        """
        bias = False if batch_norm else True
        feature_layers = [
            # 32x32 with cifar dataset
            ConvBlock(input_channel, 64, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            ResBlock(64, 64, stride=1, bias=bias, bn=batch_norm, act=act),
            ResBlock(64, 64, stride=1, bias=bias, bn=batch_norm, act=act),
            # 16x16
            ResBlock(64, 128, stride=2, bias=bias, bn=batch_norm, act=act),
            ResBlock(128, 128, stride=1, bias=bias, bn=batch_norm, act=act),
            # 8x8
            ResBlock(128, 256, stride=2, bias=bias, bn=batch_norm, act=act),
            ResBlock(256, 256, stride=1, bias=bias, bn=batch_norm, act=act),
            # 4x4
            ResBlock(256, 512, stride=2, bias=bias, bn=batch_norm, act=act),
            ResBlock(512, 512, stride=1, bias=bias, bn=batch_norm, act=act)
        ]
        super().__init__(num_classes, feature_layers, [ic_index, -1])


class ResNet34(SDN):
    def __init__(self, ic_index, act, num_classes, batch_norm, input_channel):
        """
        ic_index: [0, 1, 2, 3, 4, 5, ..., 13, 14, 15]
        """
        bias = False if batch_norm else True
        feature_layers = [
            # 32x32 with cifar dataset
            ConvBlock(input_channel, 64, 3, stride=1, padding=1, bias=bias, bn=batch_norm, act=act),
            ResBlock(64, 64, stride=1, bias=bias, bn=batch_norm, act=act),
            ResBlock(64, 64, stride=1, bias=bias, bn=batch_norm, act=act),
            ResBlock(64, 64, stride=1, bias=bias, bn=batch_norm, act=act),
            # 16x16
            ResBlock(64, 128, stride=2, bias=bias, bn=batch_norm, act=act),
            ResBlock(128, 128, stride=1, bias=bias, bn=batch_norm, act=act),
            ResBlock(128, 128, stride=1, bias=bias, bn=batch_norm, act=act),
            ResBlock(128, 128, stride=1, bias=bias, bn=batch_norm, act=act),
            # 8x8
            ResBlock(128, 256, stride=2, bias=bias, bn=batch_norm, act=act),
            ResBlock(256, 256, stride=1, bias=bias, bn=batch_norm, act=act),
            ResBlock(256, 256, stride=1, bias=bias, bn=batch_norm, act=act),
            ResBlock(256, 256, stride=1, bias=bias, bn=batch_norm, act=act),
            ResBlock(256, 256, stride=1, bias=bias, bn=batch_norm, act=act),
            ResBlock(256, 256, stride=1, bias=bias, bn=batch_norm, act=act),
            # 4x4
            ResBlock(256, 512, stride=2, bias=bias, bn=batch_norm, act=act),
            ResBlock(512, 512, stride=1, bias=bias, bn=batch_norm, act=act),
            ResBlock(512, 512, stride=1, bias=bias, bn=batch_norm, act=act)
        ]
        super().__init__(num_classes, feature_layers, [ic_index, -1])


class ResNet50(SDN):
    def __init__(self, ic_index, act, num_classes, input_channel):
        """
        ic_index: [0, 1, 2, 3, 4, 5, ..., 15]
        """
        feature_layers = [
            # 32x32 with cifar dataset
            ConvBlock(input_channel, 64, 3, stride=1, padding=1, bias=True, bn=True, act=act),
            ResBlock2(64, 64, 256, stride=1, act=act),
            ResBlock2(256, 64, 256, stride=1, act=act),
            ResBlock2(256, 64, 256, stride=1, act=act),
            # 16x16
            ResBlock2(256, 128, 512, stride=2, act=act),
            ResBlock2(512, 128, 512, stride=1, act=act),
            ResBlock2(512, 128, 512, stride=1, act=act),
            ResBlock2(512, 128, 512, stride=1, act=act),
            # 8x8
            ResBlock2(512, 256, 1024, stride=2, act=act),
            ResBlock2(1024, 256, 1024, stride=1, act=act),
            ResBlock2(1024, 256, 1024, stride=1, act=act),
            ResBlock2(1024, 256, 1024, stride=1, act=act),
            ResBlock2(1024, 256, 1024, stride=1, act=act),
            ResBlock2(1024, 256, 1024, stride=1, act=act),
            # 4x4
            ResBlock2(1024, 512, 2048, stride=2, act=act),
            ResBlock2(2048, 512, 2048, stride=1, act=act),
            ResBlock2(2048, 512, 2048, stride=1, act=act)
        ]
        super().__init__(num_classes, feature_layers, [ic_index, -1])
