import torch
import monai

def normalize_over_time(x, axis=1):
    return (x - x.mean(axis=axis, keepdims=True))/(x.std(axis=axis, keepdims=True))

def velocity_divergence(x_real):
    """ Compute the divergence of the velocity of a real signal
        Equal to the divergence of the gradient of the phase of the Analytic Signal
        Shape = [B, T, N_Caps, *cap_shape]
    """
    assert len(x_real.shape) == 4
    ht = monai.networks.layers.HilbertTransform(axis=1)
    # B, T, N, D = x_real.shape
    # x_real_pad = torch.nn.functional.pad(x_real.view(-1, N, D), pad=(1,1), mode='circular').view(B, T, N, D+2)
    x_real = normalize_over_time(x_real, axis=1)
    x_a = ht(x_real)

    xp1 = x_a.roll(shifts=1, dims=-1)
    xm1 = x_a.roll(shifts=-1, dims=-1)

    vel = torch.angle(xp1 * torch.conj(xm1)) / 2.0

    # div = unwrap(torch.angle(xp1 * torch.conj(x_a))) - unwrap(torch.angle(x_a * torch.conj(xm1)))
    div = torch.angle(xp1 * torch.conj(x_a)) - torch.angle(x_a * torch.conj(xm1))
    return div, vel

def velocity_divergence_2d(x_real):
    """ Compute the divergence of the velocity of a real signal
        Equal to the divergence of the gradient of the phase of the Analytic Signal
        Shape = [B, T, N_Caps, *cap_shape]
    """
    assert len(x_real.shape) == 5
    ht = monai.networks.layers.HilbertTransform(axis=1)
    # B, T, N, D = x_real.shape
    # x_real_pad = torch.nn.functional.pad(x_real.view(-1, N, D), pad=(1,1), mode='circular').view(B, T, N, D+2)
    x_a = ht(x_real)

    x_xp1 = x_a.roll(shifts=1, dims=-1)
    x_xm1 = x_a.roll(shifts=-1, dims=-1)
    x_yp1 = x_a.roll(shifts=1, dims=-2)
    x_ym1 = x_a.roll(shifts=-1, dims=-2)

    vel_x = torch.angle(x_xp1 * torch.conj(x_xm1)) / 2.0
    vel_y = torch.angle(x_yp1 * torch.conj(x_ym1)) / 2.0

    # div = unwrap(torch.angle(xp1 * torch.conj(x_a))) - unwrap(torch.angle(x_a * torch.conj(xm1)))
    # div = torch.angle(xp1 * torch.conj(x_a)) - torch.angle(x_a * torch.conj(xm1))
    return None, torch.stack([vel_y, vel_x], dim=0)


def rgb_phase(x_real):
    ht = monai.networks.layers.HilbertTransform(axis=1)
    x_real = normalize_over_time(x_real, axis=1)
    x_anal = ht(x_real)
    phase = torch.angle(x_anal)
    phase_r = torch.cos(phase)
    phase_g = torch.cos(phase + 2.0944) 
    phase_b = torch.cos(phase - 2.0944) 
    rgb = torch.stack([phase_r, phase_g, phase_b], dim=2)
    return rgb
