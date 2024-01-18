import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model_ann import DenseBlock, ConvBlock, ResBlock, ResBlock2


MAC_ENE = 3.7 + 0.9
AC_ENE = 0.9


def create_model_snn(model, batch_size, input_shape):
    spiking_model = SpikingSDN(model, batch_size, input_shape)
    return spiking_model


class SpikingModule(torch.nn.Module):
    def __init__(self, out_shape, alpha=0.01, inputs_float=False):
        super().__init__()

        self.out_shape = out_shape
        self.alpha = alpha
        self.v_th = 1.0
        self.register_buffer('v', torch.zeros(self.out_shape))
        self.register_buffer('out', torch.zeros(self.out_shape))
        self.step = 0
        self.spikecount = 0
        self.inputs_float = inputs_float
        self.spikecount_multiplier = MAC_ENE / AC_ENE if inputs_float else AC_ENE / AC_ENE
        print(self.spikecount_multiplier)
            

    def forward(self, x, f, *args):
        self.step += 1
        prev = self.out.clone()
        if self.inputs_float:
            if self.step == 1:
                w_size = float(torch.numel(self.weight))
                mac_count = w_size * ((x.size(2)+2*self.padding[0])/self.stride[0]) * ((x.size(3)+2*self.padding[1])/self.stride[1])
                self.spikecount += mac_count * self.spikecount_multiplier + float(torch.numel(self.bias))
            else:
                self.spikecount += float(torch.numel(torch.ones(self.out_shape)))
        elif not self.inputs_float:
            self.spikecount += torch.sum(torch.abs(x))

        x = f(x, *args)

        self.v += x
        plus = (self.v >= self.v_th).float()
        if self.alpha is not None:
            minus = (self.v <= -self.v_th/self.alpha).float()
            self.v = self.v - plus + minus/self.alpha
            self.out = plus - minus
        else:
            self.v -= plus
            self.out = plus

        return prev

    def reset(self):
        self.v.zero_()
        self.spikecount = 0

    def get_count(self):
        return self.spikecount

    def set_membrane(self, mask, source):
        self.v.masked_scatter_(mask, source)


class SpikingDense(SpikingModule):
    def __init__(self, denseblock, batch_size):
        self.out_shape = (batch_size, denseblock.main.out_features)

        if isinstance(denseblock.act, nn.LeakyReLU):
            super().__init__(
                self.out_shape,
                alpha=denseblock.act.negative_slope)
        else:
            super().__init__(
                self.out_shape,
                alpha=None)

        self.weight = nn.Parameter(denseblock.main.weight)
        self.bias = nn.Parameter(denseblock.main.bias)

    def forward(self, x):
        return super().forward(x, F.linear, self.weight, self.bias)


class SpikingConv(SpikingModule):
    def __init__(self, convblock, out_shape, batch_size, inputs_float=False):
        self.out_shape = (batch_size,) + out_shape
        if isinstance(convblock.act, nn.LeakyReLU):
            super().__init__(
                self.out_shape,
                alpha=convblock.act.negative_slope,
                inputs_float=inputs_float)
        else:
            super().__init__(
                self.out_shape,
                alpha=None,
                inputs_float=inputs_float)

        self.weight = nn.Parameter(convblock.main.weight)
        self.bias = nn.Parameter(convblock.main.bias)
        self.stride = convblock.main.stride
        self.padding = convblock.main.padding
        weight_shape = convblock.main.weight.shape
        self.spikecount_multiplier = weight_shape[0] * \
            np.prod(weight_shape[2:])

    def forward(self, x):
        return super().forward(x, F.conv2d,  self.weight, self.bias, self.stride, self.padding)


class SpikingAvgPool(SpikingModule):
    def __init__(self, out_shape, kernel_size, batch_size):
        self.out_shape = (batch_size,) + out_shape
        self.kernel_size = kernel_size
        super().__init__(
            self.out_shape,
            alpha=None)

        self.kernel_size = kernel_size
        self.spikecount_multiplier = np.prod(kernel_size)

    def forward(self, x):
        return super().forward(x, F.avg_pool2d, self.kernel_size)


class SpikingResBlock(nn.Module):
    def __init__(self, resblock, out_shape, batch_size):
        super().__init__()

        self.stride = resblock.conv1.stride[0]
        mid_shape = (out_shape[0], out_shape[1] *
                     self.stride, out_shape[2]*self.stride)
        self.out_shape = (batch_size,) + out_shape

        self.conv1_weight = nn.Parameter(resblock.conv1.weight)
        self.conv1_bias = nn.Parameter(resblock.conv1.bias)
        self.conv1 = SpikingModule(self.out_shape, alpha=None)

        self.conv2 = nn.Conv2d(resblock.conv2.in_channels,
                               resblock.conv2.out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.conv2.weight = nn.Parameter(resblock.conv2.weight)
        self.conv2.bias = nn.Parameter(resblock.conv2.bias)

        self.conv_skip = nn.Conv2d(resblock.conv_skip.in_channels,
                                   resblock.conv_skip.out_channels,
                                   kernel_size=1, stride=self.stride,
                                   padding=0, bias=True)
        self.conv_skip.weight = nn.Parameter(resblock.conv_skip.weight)
        self.conv_skip.bias = nn.Parameter(resblock.conv_skip.bias)
        self.out = SpikingAvgPool(
            self.out_shape[1:], kernel_size=1, batch_size=self.out_shape[0])

    def forward(self, x):
        skip = self.conv_skip(x)

        x = self.conv1.forward(x, F.conv2d, self.conv1_weight, self.conv1_bias,
                               self.stride, 1)
        x = self.conv2(x)
        x = x + skip
        x = self.out(x)
        return x

    def reset(self):
        self.conv1.reset()
        self.out.reset()

    def get_count(self):
        return self.conv1.get_count() + self.out.get_count()
    

class SpikingResBlock2(nn.Module):
    def __init__(self, resblock, out_shape, batch_size):
        super().__init__()

        self.stride = resblock.convM.stride[0]
        self.mid_shape = (out_shape[0]//4, out_shape[1] *
                     self.stride, out_shape[2]*self.stride)
        self.out_shape = (batch_size,) + out_shape

        self.conv1_weight = nn.Parameter(resblock.conv1.weight)
        self.conv1_bias = nn.Parameter(resblock.conv1.bias)
        self.conv1 = SpikingModule(self.mid_shape, alpha=None)
        
        self.convM_weight = nn.Parameter(resblock.convM.weight)
        self.convM_bias = nn.Parameter(resblock.convM.bias)
        self.convM = SpikingModule(self.mid_shape, alpha=None)

        self.conv2 = nn.Conv2d(resblock.conv2.in_channels,
                               resblock.conv2.out_channels,
                               kernel_size=1, stride=1,
                               padding=1, bias=True)
        self.conv2.weight = nn.Parameter(resblock.conv2.weight)
        self.conv2.bias = nn.Parameter(resblock.conv2.bias)

        self.conv_skip = nn.Conv2d(resblock.conv_skip.in_channels,
                                   resblock.conv_skip.out_channels,
                                   kernel_size=1, stride=self.stride,
                                   padding=0, bias=True)
        self.conv_skip.weight = nn.Parameter(resblock.conv_skip.weight)
        self.conv_skip.bias = nn.Parameter(resblock.conv_skip.bias)
        self.out = SpikingAvgPool(
            self.out_shape[1:], kernel_size=1, batch_size=self.out_shape[0])

    def forward(self, x):
        skip = self.conv_skip(x)

        x = self.conv1.forward(x, F.conv2d, self.conv1_weight, self.conv1_bias,
                               1, 1)
        x = self.convM.forward(x, F.conv2d, self.convM_weight, self.convM_bias,
                               self.stride, 1)
        x = self.conv2(x)
        x = x + skip
        x = self.out(x)
        return x

    def reset(self):
        self.conv1.reset()
        self.convM.reset()
        self.out.reset()

    def get_count(self):
        return self.conv1.get_count() + self.convM.get_count() + self.out.get_count()


class SpikingSDN(nn.Module):
    def __init__(self, ann, batch_size, initial_shape):
        super().__init__()

        shape = initial_shape
        shapes = []

        feature_layers = []
        cnt = 0
        for module in ann.feature:
            if isinstance(module, ConvBlock):
                shape = (module.main.out_channels, shape[1], shape[2])
                if module.main.stride[0] == 2:
                    shape = (shape[0], int((shape[1]+module.main.padding[0]*2-module.main.kernel_size[0])/2)+1, int((shape[2]+module.main.padding[0]*2-module.main.kernel_size[0])/2)+1)
                feature_layers.append(SpikingConv(module, shape, batch_size, True if cnt == 0 else False))
                cnt += 1
            elif isinstance(module, ResBlock):
                shape = (module.conv1.out_channels, shape[1], shape[2])
                if module.conv1.stride[0] == 2:
                    shape = (shape[0], int((shape[1]+module.conv1.padding[0]*2-module.conv1.kernel_size[0])/2)+1, int((shape[2]+module.conv1.padding[0]*2-module.conv1.kernel_size[0])/2)+1)
                feature_layers.append(SpikingResBlock(module, shape, batch_size))
            elif isinstance(module, ResBlock2):
                shape = (module.convM.out_channels, shape[1], shape[2])
                if module.convM.stride[0] == 2:
                    shape = (shape[0], int((shape[1]+module.convM.padding[0]*2-module.convM.kernel_size[0])/2)+1, int((shape[2]+module.convM.padding[0]*2-module.convM.kernel_size[0])/2)+1)
                feature_layers.append(SpikingResBlock2(module, shape, batch_size))
            shapes.append(shape)
        self.feature = nn.ModuleList(feature_layers)

        classifiers = {}
        for index, module in ann.classifiers.items():
            shape = shapes[int(index)]
            pool_size = shape[1]
            classifiers[index] = nn.Sequential(
                SpikingAvgPool((shape[0], 1, 1), pool_size, batch_size),
                nn.Flatten(),
                SpikingDense(module[2], batch_size)
            )
        self.classifiers = nn.ModuleDict(classifiers)

    def forward(self, x):
        y = []
        for module in self.feature:
            x = module(x)
            y.append(x)

        out = {}
        for index, module in self.classifiers.items():
            out[int(index)] = module(y[int(index)])

        return out

    def reset(self):
        for module in self.feature:
            module.reset()

        for module in self.classifiers.values():
            module[0].reset()
            module[2].reset()
            
    def set_vth(self, vth):
        for module in self.feature:
            module.v_th = vth
            
        for module in self.classifiers.values():
            module[2].v_th = vth

    def get_count(self, ic_index):
        count = 0
        count_mid = 0
        count_final = 0
        for index, module in enumerate(self.feature):
            count += module.get_count()
            count_final += module.get_count()
            if index < ic_index:
              count_mid += module.get_count()

        for index, module in self.classifiers.items():
            index = int(index)
            count += module[0].get_count() + module[2].get_count()
            if index == ic_index:
              count_mid += module[0].get_count() + module[2].get_count()
            if index == -1:
              count_final += module[0].get_count() + module[2].get_count()


        return count, count_mid, count_final