import torch
import numpy as np
import wandb

def get_batch(T,batch_size):
    values = torch.rand(T, batch_size, requires_grad=False)
    indices = torch.zeros_like(values)
    half = int(T / 2)
    for i in range(batch_size):
        half_1 = np.random.randint(half)
        hals_2 = np.random.randint(half, T)
        indices[half_1, i] = 1
        indices[hals_2, i] = 1

    data = torch.stack((values, indices), dim=-1)
    targets = torch.mul(values, indices).sum(dim=0)
    return data, targets


def normalize_int(x):
  x -= x.min()
  x *= 255 / x.max()
  return x.int()


def Plot_Vid(seq, fps=60, vformat='gif', name='Latents'):
    n_t, n_cin, nh, nw = seq.shape
    # Seq shape should be T,C,H,W

    seq_norm = normalize_int(seq).cpu()
    
    wandb_video = wandb.Video(seq_norm, fps=fps, format=vformat)
    wandb.log({name: wandb_video})
