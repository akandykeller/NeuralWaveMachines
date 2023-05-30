import torchvision
import os
import wandb 
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import seaborn as sns
from pylab import *

def plot_recon(x, xhat, s_dir, e, wandb_on, max_plot=100, extra_name=''):
    x_path = os.path.join(s_dir, f'{e}_x.png')
    xhat_path = os.path.join(s_dir, f'{e}_xrecon.png')
    diff_path = os.path.join(s_dir, f'{e}_recon_diff.png')

    xhat = xhat[:max_plot]
    x = x[:max_plot]

    n_row = int(x.shape[0] ** 0.5)

    os.makedirs(s_dir, exist_ok=True)
    torchvision.utils.save_image(
        xhat, xhat_path, nrow=n_row,
        padding=2, normalize=False)

    torchvision.utils.save_image(
        x, x_path, nrow=n_row,
        padding=2, normalize=False)

    xdiff = torch.abs(x - xhat)

    torchvision.utils.save_image(
        xdiff, diff_path, nrow=n_row,
        padding=2, normalize=False)

    if wandb_on:
        wandb.log({'X Original' + extra_name:  wandb.Image(x_path)})
        wandb.log({'X Recon' + extra_name:  wandb.Image(xhat_path)})
        wandb.log({'Recon diff' + extra_name:  wandb.Image(diff_path)})


def plot_traversal_recon(x, s_dir, e, n_transforms, wandb_on, name='Cap_Traversal'):
    x_orig = x[0][0:n_transforms]
    x_recon = x[1][0:n_transforms]
    x_trav = [t[0].unsqueeze(0) for t in x[2:]]

    x_trav_path = os.path.join(s_dir, f'{e}_{name}.png')
    os.makedirs(s_dir, exist_ok=True)

    x_image = torch.cat([x_orig, x_recon] + x_trav)

    torchvision.utils.save_image(
        x_image, x_trav_path, nrow=n_transforms,
        padding=2, pad_value=1.0, normalize=False)

    if wandb_on:
        wandb.log({name:  wandb.Image(x_trav_path)})
    

def plot_filters(weight, name='TVAE_Filters', max_s=64, max_plots=3, max_inches=5, wandb_on=True):
    weights_grid = weight.detach().cpu().numpy()
    empy_weight = np.zeros_like(weights_grid[0,0,:,:])
    c_out, c_in, h, w = weights_grid.shape
    s = min(max_s, int(np.ceil(np.sqrt(c_out))))

    if c_in == 3:
        f, axarr = plt.subplots(s,s)
        f.set_size_inches(min(s, max_inches), min(s, max_inches))
        empy_weight = np.zeros_like(weights_grid[0,:,:,:]).transpose((1, 2, 0))
        for s_h in range(s):
            for s_w in range(s):
                w_idx = s_h * s + s_w
                if w_idx < c_out:
                    filter_norm = colors.Normalize()(weights_grid[w_idx, :, :, :].transpose((1, 2, 0)))
                    img = axarr[s_h, s_w].imshow(filter_norm)
                    axarr[s_h, s_w].get_xaxis().set_visible(False)
                    axarr[s_h, s_w].get_yaxis().set_visible(False)
                    # f.colorbar(img, ax=axarr[s_h, s_w])
                else:
                    img = axarr[s_h, s_w].imshow(empy_weight)
                    axarr[s_h, s_w].get_xaxis().set_visible(False)
                    axarr[s_h, s_w].get_yaxis().set_visible(False)
                    # f.colorbar(img, ax=axarr[s_h, s_w])
        if wandb_on:
            wandb.log({"{}".format(name): wandb.Image(plt)}, commit=False)
        else:
            plt.savefig("{}".format(name))
        plt.close('all')
    else:
        for c in range(min(c_in, max_plots)):
            f, axarr = plt.subplots(s,s)
            f.set_size_inches(min(s, max_inches), min(s, max_inches))
            for s_h in range(s):
                for s_w in range(s):
                    w_idx = s_h * s + s_w
                    if w_idx < c_out:
                        img = axarr[s_h, s_w].imshow(weights_grid[w_idx, c, :, :], cmap='PuBu_r')
                        axarr[s_h, s_w].get_xaxis().set_visible(False)
                        axarr[s_h, s_w].get_yaxis().set_visible(False)
                        # f.colorbar(img, ax=axarr[s_h, s_w])
                    else:
                        img = axarr[s_h, s_w].imshow(empy_weight, cmap='PuBu_r')
                        axarr[s_h, s_w].get_xaxis().set_visible(False)
                        axarr[s_h, s_w].get_yaxis().set_visible(False)
                        # f.colorbar(img, ax=axarr[s_h, s_w])

            if wandb_on:
                wandb.log({"{}_cin{}".format(name, c): wandb.Image(plt)}, commit=False)
            else:
                plt.savefig("{}".format(name))
            plt.close('all')

def Plot_MaxActImg(all_s, all_x, s_dir, e, wandb_on):
    max_xs = []
    for s_idx in range(all_s.shape[1]):
        max_idx = torch.max(torch.abs(all_s[:, s_idx]), 0)[1]
        max_xs.append(all_x[max_idx].squeeze().unsqueeze(0).unsqueeze(0))
    
    path = os.path.join(s_dir, f'{e}_maxactimg.png')
    os.makedirs(s_dir, exist_ok=True)

    x_image = torch.cat(max_xs)

    sq = int(float(all_s.shape[1]) ** 0.5)

    torchvision.utils.save_image(
        x_image, path, nrow=sq,
        padding=2, normalize=False)

    if wandb_on:
        wandb.log({'Max_Act_Img':  wandb.Image(path)})

def normalize_int(x):
  x -= x.min()
  x *= 255 / x.max()
  return x.int()


def Plot_Capsule_Vid(seq, n_caps, cap_shape, n_samples=1,
                    fps=4, vformat='gif', wandb_on=True, name='Latents'):
    bsz, n_t, n_cin = seq.shape[0], seq.shape[1], seq.shape[2]
    # Seq shape should be T,C,H,W

    if n_caps == 1 and len(cap_shape) == 1:
        n_plots = 1
        n_rows = int(np.sqrt(cap_shape[0]))
        assert cap_shape[0] == n_rows ** 2.0
        n_cols = n_rows
        name += "_wrapped"
    elif n_caps > 1 and len(cap_shape) == 1:
        n_plots = 1
        n_rows = n_caps
        n_cols = cap_shape[0]
    elif len(cap_shape) == 2:
        n_plots = n_caps
        n_rows = cap_shape[0]
        n_cols = cap_shape[1]
        name += "_2D"
    else:
        raise NotImplementedError

    seq = seq.view(bsz, n_t, n_cin, n_plots, n_rows, n_cols)

    name = name + '_b{}_p{}'

    for b in range(n_samples):
        for p in range(n_plots):
            seq_norm = normalize_int(seq[b, :, :, p]).cpu()
            
            wandb_video = wandb.Video(seq_norm, fps=fps, format=vformat)
            if wandb_on:
                wandb.log({name.format(b, p): wandb_video})
            plt.close('all')

def Plot_Vector_Vid(seq, n_caps, cap_shape, n_samples=1,
                    fps=4, vformat='gif', wandb_on=True, name='Latents'):
    dims, bsz, n_t, n_caps, nh, nw = seq.shape[0], seq.shape[1], seq.shape[2], seq.shape[3], seq.shape[4], seq.shape[5]
    # Seq shape should be T,C,H,W

    if n_caps == 1 and len(cap_shape) == 1:
        n_plots = 1
        n_rows = int(np.sqrt(cap_shape[0]))
        assert cap_shape[0] == n_rows ** 2.0
        n_cols = n_rows
        name += "_wrapped"
    elif n_caps > 1 and len(cap_shape) == 1:
        n_plots = 1
        n_rows = n_caps
        n_cols = cap_shape[0]
    elif len(cap_shape) == 2:
        n_plots = n_caps
        n_rows = cap_shape[0]
        n_cols = cap_shape[1]
        name += "_2D"
    else:
        raise NotImplementedError

    seq_y = seq[0].view(bsz, n_t, 1, n_plots, n_rows, n_cols)
    seq_x = seq[1].view(bsz, n_t, 1, n_plots, n_rows, n_cols)

    name = name + '_b{}_p{}_t{}'
    downsample = 4
    for b in range(n_samples):
        for p in range(n_plots):
            # seq_norm = normalize_int(seq[b, :, :, p]).cpu()
            for t in range(n_t):
              vy = seq_y[b, t, 0, p, ::downsample, ::downsample]
              vx = seq_x[b, t, 0, p, ::downsample, ::downsample]
              plt.quiver(vx, vy)
              if wandb_on:
                wandb.log({name.format(b, p, t): wandb.Image(plt)})
              plt.close('all')



def Plot_MaxActIdx(seq, name, seq_len=18, mod_len=9):
    # Inshape = B, T, C, H, W
    if len(seq.shape) == 4:
        seq = seq.unsqueeze(2)
    n_periods = seq.shape[1] // seq_len
    if n_periods > 1:
        seq = seq.view(seq.shape[0], n_periods, 1, seq_len, seq.shape[3], seq.shape[4])
        seq = seq[:, 1:]
        n_periods = n_periods - 1
        seq = seq.view(seq.shape[0], n_periods*seq_len, 1, seq.shape[4], seq.shape[5])
    seq = seq.reshape(seq.shape[0] * n_periods, 1, seq_len, seq.shape[3], seq.shape[4])
    max_idx = torch.mode(torch.max(seq, dim=2)[1].to(float), dim=0)[0]
    phase = ((max_idx % mod_len) / float(mod_len)) * 2 * np.pi
    phase_r = torch.cos(phase)
    phase_g = torch.cos(phase + 2.0944) 
    phase_b = torch.cos(phase - 2.0944) 
    rgb = torch.stack([phase_r, phase_g, phase_b], dim=1)
    # hsv = torch.stack([phase, torch.ones_like(phase), torch.ones_like(phase)], dim=1)
    # rgb = kornia.color.hsv_to_rgb(hsv)
    wandb.log({name: wandb.Image(rgb)})
