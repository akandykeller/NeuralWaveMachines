from tvae.utils.locally_connected import LocallyConnected2d
from pkg_resources import require
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def get_conv(in_dim, out_dim, kernel_size, stride, padding, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    c = nn.Conv2d(int(in_dim), int(out_dim), kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c

def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups, scaled=scaled)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups, scaled=scaled)


class Block(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, residual=False, use_3x3=True, zero_last=False):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        # self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        # xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.log_weight)
        nn.init.zeros_(self.log_weight)
        with torch.no_grad():
            self.log_weight += torch.rand_like(self.log_weight) * -10
            self.log_weight -= torch.diag_embed(torch.diag(self.log_weight))

    def forward(self, input):
        return nn.functional.linear(input, self.log_weight.exp())

    def get_weights(self):
        return self.log_weight.exp()

    def normalize_weights(self):
        with torch.no_grad():
            self.log_weight -= torch.log(self.log_weight.exp().pow(2.0).sum(dim=1).pow(0.5).unsqueeze(-1))



class Conv1d_Flat(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv1d_Flat, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, input):
        i_c = input.unsqueeze(1)
        o_c = self.conv(i_c)
        o = o_c.squeeze(1)
        return o


class Conv_Cap2Cap(nn.Module):
    def __init__(self, n_caps, cap_shape, kernel_size, stride, padding, external_padding=0,
                 bias=False, mult_init=1.0, share_over_caps=True, forward_only=False, locally_connected=False,
                 init=None):
        super(Conv_Cap2Cap, self).__init__()
        assert len(kernel_size) == len(cap_shape) and len(kernel_size) <= 3
        self.n_cap_dim = len(kernel_size)
        self.cap_shape = cap_shape
        self.padded_cap_shape = tuple(map(lambda x: x + 2 * external_padding, cap_shape))
        self.n_caps = n_caps
        self.share_over_caps = share_over_caps
        self.forward_only = forward_only
        self.locally_connected = locally_connected

        if share_over_caps:
            self.in_channels = 1
            self.out_channels = 1 
            groups = 1
        else:
            self.in_channels = n_caps
            self.out_channels = n_caps
            groups = n_caps
            
        if self.n_cap_dim == 1 and locally_connected == False:
            self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, 
                                  kernel_size=kernel_size, stride=stride, padding=padding,
                                  groups=groups, bias=bias)
        elif self.n_cap_dim == 2 and locally_connected == False:
            self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                                  kernel_size=kernel_size, stride=stride, padding=padding,
                                  groups=groups, bias=bias)
        elif self.n_cap_dim == 3 and locally_connected == False:
            self.conv = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, 
                                  kernel_size=kernel_size, stride=stride, padding=padding,
                                  groups=groups, bias=bias)
        elif self.n_cap_dim == 2 and locally_connected == True:
            self.conv = LocallyConnected2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                                           output_size=cap_shape, kernel_size=kernel_size[0], stride=1, 
                                           padding=0, padding_type='constant', bias=False)
        
        else: 
            raise NotImplementedError

        if init is None:
            nn.init.constant_(self.conv.weight, 1.0/np.prod(cap_shape))
        else:
            init(self.conv.weight)

        if self.forward_only:
            assert self.n_cap_dim == 1
            if not share_over_caps:
                n_filts = self.conv.weight.shape[0]
            else:
                n_filts = 1
            self.mult = nn.Parameter(torch.Tensor([mult_init]).repeat((n_filts,1)), requires_grad=True)
            self.conv.weight = nn.Parameter(torch.Tensor([[[0., 0., 1.]]]).repeat((n_filts, 1, 1)), requires_grad=False)
            # self.conv.weight = nn.Parameter(torch.Tensor([[[1., 0., 1.]]]), requires_grad=False)
            if bias:
                nn.init.constant_(self.conv.bias, 0.0)
        else:
            self.mult = 1.0 # mult_init

        # self.conv.weight.requires_grad = False

    def forward(self, x):
        x_c = x.reshape(-1, self.in_channels, *self.padded_cap_shape)
        o_c = self.mult * self.conv(x_c)
        o = o_c.reshape(x.shape[0], self.n_caps, *self.cap_shape)
        return o

class Linear_Cap2Cap(nn.Module):
    def __init__(self, n_caps, cap_shape, bias=False, share_over_caps=True):
        super(Linear_Cap2Cap, self).__init__()
        self.cap_shape = cap_shape
        self.n_caps = n_caps
        self.share_over_caps = share_over_caps
        
        if share_over_caps:
            self.n_features = np.prod(cap_shape)
        else:
            self.n_features = np.prod(cap_shape) * n_caps

        self.linear = nn.Linear(in_features=self.n_features, out_features=self.n_features)


    def forward(self, x):
        x_r = x.reshape(-1, self.n_features)
        o = self.linear(x_r)
        o = o.reshape(x.shape[0], self.n_caps, *self.cap_shape)
        return o


class Linear_In2Cap(nn.Module):
    def __init__(self, input_size, n_caps, cap_shape):
        super(Linear_In2Cap, self).__init__()
        self.n_caps = n_caps
        self.cap_shape = cap_shape
        self.linear = nn.Linear(in_features=input_size, out_features=n_caps * np.prod(cap_shape))
    
    def forward(self, x):
        output = self.linear(x)
        output = output.reshape(x.shape[0], self.n_caps, *self.cap_shape)
        return output