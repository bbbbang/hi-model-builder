import torch
import torch.nn as nn



def fuse_conv_bn(conv, bn):
    kernel = conv.weight
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    running_mean = bn.running_mean
    running_var = bn.running_var

    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)

    return kernel * t, beta - running_mean * gamma / std



class ConvBnNonlinear(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                nonlinear=None) -> None:
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.nonlinear= nonlinear

    def forward(self, x):
        return self.nonlinear(self.bn(self.conv(x)))
    
    def fuse(self):

        new_weight, new_bias = fuse_conv_bn(self.conv, self.bn)

        new_conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels,
                                    kernel_size=self.conv.kernel_size, stride=self.conv.stride,
                                    padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups, bias=True)
        new_conv.weight.data = new_weight
        new_conv.bias.data = new_bias

        setattr(self, "conv", new_conv)
        setattr(self, "bn", nn.Identity())

