import torch
import numpy as np
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from tvae.utils.vis import plot_filters
from tvae.utils.locally_connected import LocallyConnected2d, LocallyConnectedTransposed2d

class Decoder(nn.Module):
    def __init__(self, model):
        super(Decoder, self).__init__()
        self.model = model

    def forward(self, x):
        raise NotImplementedError

    def plot_weights(self, name, wandb_on=True, max_plots=3):
        name_w_idx = name + '_L{}'
        for idx, layer in enumerate(self.model):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                plot_filters(layer.weight, name_w_idx.format(idx), wandb_on=wandb_on)
            elif isinstance(layer, LocallyConnectedTransposed2d):
                # (1, out_channels, in_channels, kernel_size**2, output_size[0], output_size[1])
                _, n_cout, n_cin, prod_ksize, oh, ow = layer.weight.shape
                assert n_cout == 1
                khw = int(prod_ksize ** 0.5)
                for ic in range(min(n_cin, max_plots)):
                    print(f"Dec: {ic}")
                    w_c = layer.weight[0, 0, ic].permute(1,2,0).view(oh*ow, 1, khw, khw)                    
                    plot_filters(w_c, f'{name}_L{idx}_cout{ic}', wandb_on=wandb_on)
            elif isinstance(layer, nn.Linear):
                s_dim = layer.weight.shape[1]
                w_hw = int(np.sqrt(layer.weight.shape[0]))
                w = layer.weight.permute(1,0).view(s_dim, 1, w_hw, w_hw)
                plot_filters(w, name_w_idx.format(idx), wandb_on=wandb_on)


class Bernoulli_Decoder(Decoder):
    def __init__(self, model):
        super(Bernoulli_Decoder, self).__init__(model)

    def forward(self, z, x):
        probs_x = torch.clamp(self.model(z), 0, 1)
        p = Bernoulli(probs=probs_x, validate_args=False)
        neg_logpx_z = -1 * p.log_prob(x.view(probs_x.shape))

        return probs_x, neg_logpx_z

    def only_decode(self, z):
        probs_x = torch.clamp(self.model(z), 0, 1)
        return probs_x