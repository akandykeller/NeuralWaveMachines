import os
from tvae.utils.train_loops import eval_epoch_autoregressive, train_epoch_autoregressive
from tvae.containers.autoregressive import Topographic_Autoregressive_NoConcat
from tvae.containers.decoder import Bernoulli_Decoder
from tvae.containers.encoder import Gaussian_Encoder
from tvae.containers.grouper import NonTopographic_1d_Autoregressive
from tvae.containers.rnn import Det_CoRNN_Caps
from tvae.data.mnist import Preprocessor
from tvae.models.mlp import LinearLayer_MLP_Encoder, LinearLayer_MLP_Decoder
from tvae.utils.logging import configure_logging, get_dirs
from tvae.utils.layer_helpers import Conv_Cap2Cap

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch.nn import functional as F
import numpy as np
import wandb

def create_model(n_caps, cap_shape, group_kernel=(1,5,1), mu_init=5.0, padding_mode='circular', 
                 forward_only=False, share_over_caps=True, dt=0.0, n_internal_steps=1, gamma=1.0,
                 alpha=0.5, kernel_shape=(3,), stride=1, padding=0, n_cin=1, n_hw=28, locally_connected=False):

    x_dim = n_hw * n_hw * n_cin
    h_dim = int(n_caps * np.prod(cap_shape))
    t_dim = int(n_caps * np.prod(cap_shape)) 
    z_dim = t_dim * 2
    padding_mode = padding_mode
    external_padding = kernel_shape[0] // 2

    encoder = Gaussian_Encoder(model=LinearLayer_MLP_Encoder(in_features=x_dim, out_features=z_dim))

    decoder = Bernoulli_Decoder(model=LinearLayer_MLP_Decoder(in_features=h_dim, out_features=x_dim))

    rnn = Det_CoRNN_Caps(input_size=x_dim, n_caps=n_caps, cap_shape=cap_shape, dt=dt, gamma=gamma, alpha=alpha,
                coupling=Conv_Cap2Cap(n_caps=n_caps, cap_shape=cap_shape, kernel_size=kernel_shape, stride=stride, 
                                        padding=padding, external_padding=external_padding, mult_init=1.0, share_over_caps=share_over_caps,
                                        forward_only=forward_only, locally_connected=locally_connected),
                damping=Conv_Cap2Cap(n_caps=n_caps, cap_shape=cap_shape, kernel_size=kernel_shape, stride=stride, 
                                        padding=padding, external_padding=external_padding, mult_init=1.0, share_over_caps=share_over_caps,
                                        forward_only=forward_only, locally_connected=locally_connected),
                driving=Conv_Cap2Cap(n_caps=n_caps, cap_shape=cap_shape, kernel_size=kernel_shape, stride=stride, 
                                    padding=padding, external_padding=external_padding, mult_init=1.0, share_over_caps=share_over_caps,
                                    forward_only=forward_only, locally_connected=locally_connected),
                padder=lambda x: F.pad(x, (0, 0, external_padding, external_padding), mode=padding_mode),
                n_internal_steps=n_internal_steps, layernorm=True)

    grouper = NonTopographic_1d_Autoregressive(nn.ConvTranspose3d(in_channels=1, out_channels=1,
                                          kernel_size=group_kernel, 
                                          padding=(2*(group_kernel[0] // 2), 
                                                   2*(group_kernel[1] // 2),
                                                   2*(group_kernel[2] // 2)),
                                          stride=(1,1,1), padding_mode='zeros', bias=False),
                      lambda x: F.pad(x, (group_kernel[2] // 2, group_kernel[2] // 2,
                                          group_kernel[1] // 2, group_kernel[1] // 2,
                                          group_kernel[0] // 2, group_kernel[0] // 2), 
                                          mode=padding_mode),
                       n_caps=n_caps, cap_dim=int(np.prod(cap_shape)),
                       mu_init=mu_init)

    return Topographic_Autoregressive_NoConcat(encoder, decoder, rnn, grouper, ic_encoder=None, steps_ahead=1)


def main():
    config = {
        'wandb_on': True,
        'lr': 2.5e-4,
        'momentum': 0.9,
        'batch_size': 8,
        'max_epochs': 500,
        'eval_epochs': 5,
        'dataset': 'MNIST',
        'train_angle_set': '0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340 ',
        'test_angle_set': '0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340 ', 
        'train_color_set':'0',
        'test_color_set': '0',
        'train_scale_set': '1.0',
        'test_scale_set': '1.0', 
        'pct_val': 0.2,
        'random_crop': 28,
        'seed': 1,
        'n_is_samples': 10,
        'cap_shape': (54,),
        'kernel_shape': (3,),
        'n_caps': 24,
        'coupling_type': 'conv', 
        'forward_only': True,
        'dt': -1.95,
        'gamma': 1.0, 
        'alpha': 0.5,
        'eval_batches': 100,
        'share_over_caps': True,
        'periods': 1,
        'test_periods': 1,
        'padding_mode': 'circular',
        'locally_connected': False,
        }
    name = 'TCoRNN-1D_Rot-MNIST'

    config['savedir'], config['data_dir'], config['wandb_dir'] = get_dirs()

    savepath = os.path.join(config['savedir'], name)

    log, checkpoint_path = configure_logging(config, name, model=None)
    config = wandb.config

    config.update({'train_angle_set' : (config['train_angle_set']*config['periods']).strip()}, allow_val_change=True)
    config.update({'test_angle_set' : (config['test_angle_set']*config['test_periods']).strip()}, allow_val_change=True)

    preprocessor = Preprocessor(config)
    train_loader, val_loader, test_loader = preprocessor.get_dataloaders(batch_size=config['batch_size'])

    if len(config['train_color_set']) == 1:
        n_cin = 1 
    else:
        n_cin = 3

    model = create_model(n_caps=config['n_caps'], cap_shape=config['cap_shape'], 
                        padding_mode=config['padding_mode'],
                        forward_only=config['forward_only'],
                        dt=config['dt'],
                        gamma=config['gamma'], alpha=config['alpha'],
                        kernel_shape=config['kernel_shape'],
                        n_cin=n_cin, n_hw=28,
                        share_over_caps=config['share_over_caps'],
                        locally_connected=config['locally_connected'])
    model.to('cuda')

    # load_checkpoint_path = ''
    # model.load_state_dict(torch.load(load_checkpoint_path))

    optimizer = optim.SGD(model.parameters(), 
                           lr=config['lr'],
                           momentum=config['momentum'])
    scheduler = StepLR(optimizer, step_size=1, gamma=1.0)


    for e in range(config['max_epochs']):
        log('Epoch', e)

        (total_loss, total_neg_logpx_z, total_kl, 
         num_batches) = train_epoch_autoregressive(model, optimizer, 
                                                    train_loader, log,
                                                    savepath, e, eval_batches=config['eval_batches'],
                                                    plot_waves=True,
                                                    wandb_on=config['wandb_on'],
                                                    plot_forward_roll=True)

        log("Epoch Avg Loss", total_loss / num_batches)
        log("Epoch Avg -LogP(x|z)", total_neg_logpx_z / num_batches)
        log("Epoch Avg KL", total_kl / num_batches)
        scheduler.step()
        
        torch.save(model.state_dict(), checkpoint_path)

        if e % config['eval_epochs'] == 0:
            total_loss, total_neg_logpx_z, total_kl, total_is_estimate, num_batches = eval_epoch_autoregressive(model, test_loader, log, savepath, e, 
                                                                                                                n_is_samples=config['n_is_samples'],
                                                                                                                wandb_on=config['wandb_on'],
                                                                                                                plot_forward_roll=True, 
                                                                                                                plot_waves=True)
            log("Val Avg Loss", total_loss / num_batches)
            log("Val Avg -LogP(x|z)", total_neg_logpx_z / num_batches)
            log("Val Avg KL", total_kl / num_batches)
            log("Val IS Estiamte", total_is_estimate / num_batches)


if __name__ == '__main__':
    main()
