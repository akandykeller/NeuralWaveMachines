import torch
import torch.nn as nn
import torch.nn.functional as F
from tvae.utils.vis import plot_filters
import matplotlib.pyplot as plt
import wandb
from tvae.utils.layer_helpers import PositiveLinear

class Grouper(nn.Module):
    def __init__(self, model, padder):
        super(Grouper, self).__init__()
        self.model = model
        self.padder = padder

    def forward(self, z, u):
        raise NotImplementedError

    def plot_weights(self, name, wandb_on=True, max_plots=3):
        if isinstance(self.model, nn.Conv3d):
            plot_filters(self.model.weight, name, wandb_on=wandb_on)
        elif isinstance(self.model, PositiveLinear):
            weight = self.model.get_weights()
            plt.imshow(weight.detach().cpu().numpy())
            if wandb_on:
                wandb.log({f"{name}": wandb.Image(plt)}, commit=False)
            else:
                plt.savefig(f"{name}")
        else:
            raise NotImplementedError

class NonTopographic_1d_Autoregressive(Grouper):
    def __init__(self, model, padder, n_caps, cap_dim,
                 trainable=False, mu_init=1, eps=1e-6):
        super(NonTopographic_1d_Autoregressive, self).__init__(model, padder)
        self.n_caps = n_caps
        self.cap_dim = cap_dim
        self.z_ln = nn.LayerNorm([self.n_caps * self.cap_dim], elementwise_affine=False)

    def forward(self, z, u):
        z = self.z_ln(z)
        s = z 
        return s