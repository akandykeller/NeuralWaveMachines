import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class LocallyConnected2d(nn.Module):
    """
    Adapted from: https://discuss.pytorch.org/t/locally-connected-layers/26979
    """
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, padding=0, padding_type='constant', bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padder = lambda x: F.pad(x, (padding, padding, padding, padding), mode=padding_type)
        
        torch.nn.init.xavier_uniform_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = self.padder(x).unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class LocallyConnectedTransposed2d(nn.Module):
    """
    Adapted from: https://discuss.pytorch.org/t/locally-connected-layers/26979
    """
    def __init__(self, in_channels, out_channels, fm_size, output_size, kernel_size, stride, padding=0, bias=False):
        super(LocallyConnectedTransposed2d, self).__init__()
        fm_size = _pair(fm_size)
        output_size = _pair(output_size)

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size**2, fm_size[0], fm_size[1])
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.folder = torch.nn.Fold(output_size=output_size, kernel_size=kernel_size, dilation=1, padding=padding, stride=stride)

        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

        # self.padder = lambda x: F.pad(x, (padding, padding, padding, padding), mode=padding_type)
        
    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, 1, c, 1, h, w)

        # Sum in in_channel
        out = (x * self.weight).sum([2]) # (b, c_out, kernel**2, h, w)
        out = out.view(b, -1, h*w)
        out = self.folder(out)
        if self.bias is not None:
            out += self.bias
        return out