import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Det_CoRNN_Caps(nn.Module):
    def __init__(self, input_size, n_caps, cap_shape, dt, gamma, alpha,
                coupling, damping, driving, padder, sigma=nn.Tanh(),
                train_dt=True, train_gamma=True, train_alpha=True, n_internal_steps=1,
                layernorm=False):
        super(Det_CoRNN_Caps, self).__init__()
        self.dt = nn.Parameter(torch.Tensor([dt]), requires_grad=train_dt)
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=train_gamma)
        self.alpha = nn.Parameter(torch.Tensor([alpha]), requires_grad=train_alpha)
        
        ## If not sharing bias over capsules uncomment below
        # self.b_bar = nn.Parameter(torch.zeros(n_caps, *cap_shape), requires_grad=True)
        self.b_bar = nn.Parameter(torch.zeros(1, *cap_shape), requires_grad=True)  
        
        self.input_size = input_size
        self.n_caps = n_caps
        self.cap_shape = cap_shape
        self.hidden_size = int(n_caps * np.prod(cap_shape))
        self.init_cond_size = self.hidden_size * 2
        self.n_internal_steps = n_internal_steps

        self.coupling = coupling
        self.damping = damping
        self.driving = driving
        self.sigma = sigma
        self.padder = padder

        if layernorm:
            self.ln_h = nn.LayerNorm(normalized_shape=(self.n_caps, *self.cap_shape))
        else:
            self.ln_h = lambda x: x

    def forward(self, h_0, z_0):
        z_flat = z_0.view(z_0.shape[0], self.n_caps, *self.cap_shape)
        z_flat_pad = self.padder(z_flat)

        h_tm1 = h_0[:, :self.hidden_size].view(h_0.shape[0], self.n_caps, *self.cap_shape)
        v_tm1 = h_0[:, self.hidden_size:].view(h_0.shape[0], self.n_caps, *self.cap_shape)
            
        for s in range(self.n_internal_steps):    
            h_tm1_pad = self.padder(h_tm1)
            v_tm1_pad = self.padder(v_tm1)

            v_t = v_tm1 + torch.sigmoid(self.dt) * (self.sigma(
                                                    self.coupling(h_tm1_pad) 
                                                    + self.damping(v_tm1_pad)
                                                    + self.driving(z_flat_pad)
                                                    + self.b_bar
                                                    )
                                        - F.relu(self.gamma) * h_tm1
                                        - F.relu(self.alpha) * v_tm1)
            h_t = self.ln_h(h_tm1 + torch.sigmoid(self.dt) * v_t)
            h_tm1 = h_t
            v_tm1 = v_t

        h_v = torch.cat([h_t.flatten(start_dim=1), v_t.flatten(start_dim=1)], dim=-1)
        return h_v

    def preprocess_for_decoder(self, h):
        return h[:, :self.hidden_size]
