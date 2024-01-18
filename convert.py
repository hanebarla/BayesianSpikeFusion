import torch
import torch.nn as nn
import numpy as np

from model_ann import DenseBlock, ConvBlock, ResBlock, ResBlock2

def merge_and_normalize_dense(dense, bn, scale_factor, prev_factor):
    w = dense.weight
    b = dense.bias

    if bn is None:
        w = w * prev_factor / scale_factor
        b = b / scale_factor
    else:
        mean = bn.running_mean
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        beta = bn.weight
        gamma = bn.bias

        w = w * (beta / var_sqrt).reshape([dense.out_features, 1]) * prev_factor / scale_factor
        b = ((b - mean)/var_sqrt * beta + gamma) / scale_factor

    new_dense = nn.Linear(dense.in_features,
                          dense.out_features,
                          bias=True)
    new_dense.weight = nn.Parameter(w)
    new_dense.bias = nn.Parameter(b)
    return new_dense

def merge_and_normalize_conv(conv, bn, scale_factor, prev_factor):
    w = conv.weight
    b = conv.bias
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    beta = bn.weight
    gamma = bn.bias

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1]) * prev_factor / scale_factor
    b = ((b - mean)/var_sqrt * beta + gamma) / scale_factor

    new_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True)
    new_conv.weight = nn.Parameter(w)
    new_conv.bias = nn.Parameter(b)
    return new_conv

def normalize(ann, data, percentile=99.9, initial_scale_factor=1.0):
    ann.eval()
    scale_factors = []
    outputs = []
    with torch.no_grad():
        x_orig = data
        prev_factor = initial_scale_factor

        for i, (name, module) in enumerate(ann._modules.items()):
            if isinstance(module, nn.AvgPool2d) or isinstance(module, nn.AdaptiveAvgPool2d):
                x_orig = module(x_orig)

            if isinstance(module, DenseBlock):
                x_orig = x_orig.view(x_orig.size(0), -1)
                x_orig = module(x_orig)
                scale_factor = np.percentile(x_orig.cpu().detach().numpy(), percentile)
                module.main = merge_and_normalize_dense(module.main, module.bn, scale_factor, prev_factor)
                module.bn = None
                prev_factor = scale_factor
                # print(scale_factor)
                scale_factors.append(scale_factor)
                outputs.append(x_orig)

            elif isinstance(module, ConvBlock):
                x_orig = module(x_orig)
                scale_factor = np.percentile(x_orig.cpu().detach().numpy(), percentile)
                module.main = merge_and_normalize_conv(module.main, module.bn, scale_factor, prev_factor)
                module.bn = None
                prev_factor = scale_factor
                # print(scale_factor)
                scale_factors.append(scale_factor)
                outputs.append(x_orig)

            elif isinstance(module, ResBlock):
                x_1 = module.act1(module.bn1(module.conv1(x_orig)))
                x_orig = module(x_orig)
                scale_factor1 = np.percentile(x_1.cpu().detach().numpy(), percentile)
                scale_factor2 = np.percentile(x_orig.cpu().detach().numpy(), percentile)
                module.conv1 = merge_and_normalize_conv(module.conv1, module.bn1, scale_factor1, prev_factor)
                module.bn1 = None
                module.conv2 = merge_and_normalize_conv(module.conv2, module.bn2, scale_factor2, scale_factor1)
                module.bn2 = None
                module.conv_skip = merge_and_normalize_conv(module.conv_skip, nn.BatchNorm2d(module.conv_skip.out_channels), scale_factor2, prev_factor)
                prev_factor = scale_factor2
                # print(scale_factor2, scale_factor1)
                scale_factors.append(scale_factor2)
                outputs.append(x_orig)

            elif isinstance(module, ResBlock2):
                x_1 = module.act1(module.bn1(module.conv1(x_orig)))
                x_M = module.actM(module.bnM(module.convM(x_1)))
                x_orig = module(x_orig)
                scale_factor1 = np.percentile(x_1.cpu().detach().numpy(), percentile)
                scale_factorM = np.percentile(x_M.cpu().detach().numpy(), percentile)
                scale_factor2 = np.percentile(x_orig.cpu().detach().numpy(), percentile)
                module.conv1 = merge_and_normalize_conv(module.conv1, module.bn1, scale_factor1, prev_factor)
                module.bn1 = None
                module.convM = merge_and_normalize_conv(module.convM, module.bnM, scale_factorM, scale_factor1)
                module.bnM = None
                module.conv2 = merge_and_normalize_conv(module.conv2, module.bn2, scale_factor2, scale_factorM)
                module.bn2 = None
                module.conv_skip = merge_and_normalize_conv(module.conv_skip, nn.BatchNorm2d(module.conv_skip.out_channels), scale_factor2, prev_factor)
                prev_factor = scale_factor2
                # print(scale_factor2, scale_factor1)
                scale_factors.append(scale_factor2)
                outputs.append(x_orig)

    return scale_factors, outputs
