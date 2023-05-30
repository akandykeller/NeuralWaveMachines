import os
import torch
from tvae.utils.vis import Plot_Capsule_Vid, Plot_MaxActIdx, plot_recon, Plot_Vector_Vid
from tvae.utils.phase import rgb_phase, velocity_divergence_2d, normalize_over_time
import numpy as np
from tqdm import tqdm
import monai

def train_epoch_autoregressive(model, optimizer, train_loader, log, savepath, epoch, eval_batches=300,
                wandb_on=True, plot_waves=False, grad_clip_norm=None, plot_forward_roll=False):
    total_loss = 0
    total_kl = 0
    total_neg_logpx_z = 0
    num_batches = 0
    
    model.train()
    for x, label in tqdm(train_loader):
        optimizer.zero_grad()
        x = x.float().to('cuda')

        seq_len = max(x.shape[1:4])
        assert np.prod(x.shape[1:4]) == seq_len

        x_seq = x.view(x.shape[0], seq_len, *x.shape[-3:])
        h_tm1 = model.init_hidden(x_seq=x_seq)
        
        seq_kl = 0.0
        seq_neg_log_px_z = 0.0
        seq_probs_x = []
        zs = []
        hs = []
        for i in range(seq_len):
            i_p1 = (i+1) % seq_len
            z_t, u_t, h_t, probs_x, kl_z, neg_log_px_z = model(x_seq[:, i], h_tm1, x_seq[:, i_p1])
            seq_kl += kl_z.sum()
            seq_neg_log_px_z += neg_log_px_z.sum()
            seq_probs_x.append(probs_x)
            zs.append(z_t.detach().cpu())
            hs.append(h_t.detach().cpu())
            h_tm1 = h_t

        seq_z = torch.stack(zs, dim=1)
        seq_h = torch.stack(hs, dim=1)

        avg_KLD = seq_kl / (x_seq.shape[0] * seq_len)
        avg_neg_logpx_z = seq_neg_log_px_z / (x_seq.shape[0] * seq_len)

        loss = avg_neg_logpx_z + avg_KLD

        ##### Phase-based divergence computation 
        ht = monai.networks.layers.HilbertTransform(axis=1)

        cap_seq_h = seq_h.view(x_seq.shape[0], seq_len, -1, model.rnn.n_caps, *model.rnn.cap_shape)
        cap_seq_z = seq_z.view(x_seq.shape[0], seq_len, model.rnn.n_caps, *model.rnn.cap_shape)
        seq_h_pos = cap_seq_h[:,:,0]
        seq_h_vel = cap_seq_h[:,:,1]

        loss.backward()
        
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()    

        total_loss += loss.detach()
        total_neg_logpx_z += avg_neg_logpx_z.detach()
        total_kl += avg_KLD.detach()
        num_batches += 1
        b_idx = epoch * len(train_loader) + num_batches

        if b_idx % eval_batches == 0:
            log('Train Total Loss', loss.detach())
            log('Train -LogP(x|z)', avg_neg_logpx_z.detach())
            log('Train KLD', avg_KLD.detach())

            if plot_forward_roll:
                x_batched = x.view(-1, *x.shape[-3:])
                forward_roll_probs_x = model.plot_forward_roll(seq_h_pos[:, -1], t_fwd=seq_h_pos.shape[1], dt=1)

                plot_recon(x_batched, 
                        forward_roll_probs_x.view(x_batched.shape), 
                        os.path.join(savepath, 'samples'),
                        b_idx, wandb_on=wandb_on,
                        extra_name='Forward Roll')

            if plot_waves:
                seq_h = seq_h.view(seq_z.shape[0], seq_len, 2, 1, -1)
                seq_z = seq_z.view(seq_z.shape[0], seq_len, 1, -1)

                from scipy.signal import butter, lfilter

                def butter_bandpass(lowcut, highcut, fs=None, order=5):
                    return butter(order, [lowcut, highcut], fs=fs, btype='band')

                def butter_bandpass_filter(data, lowcut, highcut, fs=None, order=5):
                    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
                    y = lfilter(b, a, data, axis=1)
                    return y

                seq_h = torch.tensor(butter_bandpass_filter(seq_h, lowcut=0.2, highcut=0.4, fs=None))
                seq_h_pos = torch.tensor(butter_bandpass_filter(seq_h_pos, lowcut=0.2, highcut=0.4, fs=None))

                Plot_Capsule_Vid(seq_h[:,:,0], n_caps=model.rnn.n_caps, cap_shape=model.rnn.cap_shape,
                                 name='Latents_H-Pos', wandb_on=wandb_on)
                Plot_Capsule_Vid(seq_h[:,:,1], n_caps=model.rnn.n_caps, cap_shape=model.rnn.cap_shape,
                                 name='Latents_H-Vel', wandb_on=wandb_on)
                Plot_Capsule_Vid(seq_z, n_caps=model.rnn.n_caps, cap_shape=model.rnn.cap_shape,
                                 name='Latents_Z', wandb_on=wandb_on)
                Plot_Capsule_Vid(rgb_phase(seq_h_pos), n_caps=model.rnn.n_caps, cap_shape=model.rnn.cap_shape,
                                 name='Phase_H-Pos', wandb_on=wandb_on)
                
                if len(model.rnn.cap_shape) == 2:
                    x_real = normalize_over_time(seq_h_pos, axis=1)
                    _, vel_hp_filt = velocity_divergence_2d(x_real)
              
                    avg_vel = vel_hp_filt[:,:,18:].mean(dim=2, keepdim=True)

                    Plot_Vector_Vid(-1 * vel_hp_filt, n_caps=model.rnn.n_caps, cap_shape=model.rnn.cap_shape, name='Phase Velocity', wandb_on=wandb_on)
                    Plot_Vector_Vid(-1 * avg_vel, n_caps=model.rnn.n_caps, cap_shape=model.rnn.cap_shape, name='Avg Phase Velocity', wandb_on=wandb_on)

                x_real = normalize_over_time(seq_h_pos, axis=1)
                x_anal = ht(x_real)
                phase = torch.angle(x_anal)

                Plot_MaxActIdx(seq_h_pos, name='Max Pos Seq-Idx', seq_len=18, mod_len=18)
                Plot_MaxActIdx(seq_h_vel, name='Max Vel Seq-idx', seq_len=18, mod_len=18)
                Plot_MaxActIdx(phase, name='Max Phase Seq-Idx', seq_len=18, mod_len=18)

            seq_probs_x = torch.stack(seq_probs_x, dim=1)

            x_batched = x.view(-1, *x.shape[-3:])
            plot_recon(x_batched, 
                       seq_probs_x.view(x_batched.shape), 
                       os.path.join(savepath, 'samples'),
                       b_idx, wandb_on=wandb_on)

    return total_loss, total_neg_logpx_z, total_kl, num_batches



def eval_epoch_autoregressive(model, val_loader, log, savepath, epoch, n_is_samples=100, 
                wandb_on=True,  plot_forward_roll=False, plot_waves=True, eval_batches=100):
    total_loss = 0
    total_kl = 0
    total_neg_logpx_z = 0
    total_is_estimate = 0.0
    num_batches = 0

    model.eval()
    with torch.no_grad():
        for x, label in tqdm(val_loader):
            x = x.float().to('cuda')

            seq_len = max(x.shape[1:4])
            assert np.prod(x.shape[1:4]) == seq_len

            x_seq = x.view(x.shape[0], seq_len, *x.shape[-3:])
            h_tm1 = model.init_hidden(x_seq=x_seq)
            
            seq_kl = 0.0
            seq_neg_log_px_z = 0.0
            seq_probs_x = []
            zs = []
            hs = []
            for i in range(seq_len):
                i_p1 = (i+1) % seq_len
                z_t, u_t, h_t, probs_x, kl_z, neg_log_px_z = model(x_seq[:, i], h_tm1, x_seq[:, i_p1])
                seq_kl += kl_z.sum()
                seq_neg_log_px_z += neg_log_px_z.sum()
                seq_probs_x.append(probs_x)
                zs.append(z_t.detach().cpu())
                hs.append(h_t.detach().cpu())
                h_tm1 = h_t

            seq_z = torch.stack(zs, dim=1)
            seq_h = torch.stack(hs, dim=1)

            ##### Phase-based divergence computation 
            ht = monai.networks.layers.HilbertTransform(axis=1)

            cap_seq_h = seq_h.view(x_seq.shape[0], seq_len, -1, model.rnn.n_caps, *model.rnn.cap_shape)
            cap_seq_z = seq_z.view(x_seq.shape[0], seq_len, model.rnn.n_caps, *model.rnn.cap_shape)
            seq_h_pos = cap_seq_h[:,:,0]
            seq_h_vel = cap_seq_h[:,:,1]

            num_batches += 1
            b_idx = epoch * len(val_loader) + num_batches

            if b_idx % eval_batches == 0:
                if plot_forward_roll:
                    x_batched = x.view(-1, *x.shape[-3:])
                    forward_roll_probs_x = model.plot_forward_roll(seq_h_pos[:, -1], t_fwd=seq_h_pos.shape[1], dt=1)

                    plot_recon(x_batched, 
                            forward_roll_probs_x.view(x_batched.shape), 
                            os.path.join(savepath, 'samples'),
                            epoch, wandb_on=wandb_on,
                            extra_name='Val Forward Roll')

                if plot_waves:
                    seq_h = seq_h.view(seq_z.shape[0], seq_len, 2, 1, -1)
                    seq_z = seq_z.view(seq_z.shape[0], seq_len, 1, -1)

                    from scipy.signal import butter, lfilter

                    def butter_bandpass(lowcut, highcut, fs=None, order=5):
                        return butter(order, [lowcut, highcut], fs=fs, btype='band')

                    def butter_bandpass_filter(data, lowcut, highcut, fs=None, order=5):
                        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
                        y = lfilter(b, a, data, axis=1)
                        return y

                    seq_h = torch.tensor(butter_bandpass_filter(seq_h, lowcut=0.2, highcut=0.4, fs=None))
                    seq_h_pos = torch.tensor(butter_bandpass_filter(seq_h_pos, lowcut=0.2, highcut=0.4, fs=None))

                    Plot_Capsule_Vid(seq_h[:,:,0], n_caps=model.rnn.n_caps, cap_shape=model.rnn.cap_shape,
                                    name='Val Latents_H-Pos', wandb_on=wandb_on)
                    Plot_Capsule_Vid(seq_h[:,:,1], n_caps=model.rnn.n_caps, cap_shape=model.rnn.cap_shape,
                                    name='Val Latents_H-Vel', wandb_on=wandb_on)
                    Plot_Capsule_Vid(seq_z, n_caps=model.rnn.n_caps, cap_shape=model.rnn.cap_shape,
                                    name='Val Latents_Z', wandb_on=wandb_on)
                    Plot_Capsule_Vid(rgb_phase(seq_h_pos), n_caps=model.rnn.n_caps, cap_shape=model.rnn.cap_shape,
                                    name='Val Phase_H-Pos', wandb_on=wandb_on)
                    
                    if len(model.rnn.cap_shape) == 2:
                        x_real = normalize_over_time(seq_h_pos, axis=1)
                        _, vel_hp_filt = velocity_divergence_2d(x_real)
                
                        avg_vel = vel_hp_filt[:,:,18:].mean(dim=2, keepdim=True)

                        Plot_Vector_Vid(vel_hp_filt, n_caps=model.rnn.n_caps, cap_shape=model.rnn.cap_shape, name='Val Phase Velocity', wandb_on=wandb_on)
                        Plot_Vector_Vid(avg_vel, n_caps=model.rnn.n_caps, cap_shape=model.rnn.cap_shape, name='Val Avg Phase Velocity', wandb_on=wandb_on)

                seq_probs_x = torch.stack(seq_probs_x, dim=1)

                x_batched = x.view(-1, *x.shape[-3:])
                plot_recon(x_batched, 
                        seq_probs_x.view(x_batched.shape), 
                        os.path.join(savepath, 'samples'),
                        epoch, wandb_on=wandb_on,
                        extra_name='Val')

            avg_KLD = seq_kl / (x_seq.shape[0] * seq_len)
            avg_neg_logpx_z = seq_neg_log_px_z / (x_seq.shape[0] * seq_len)

            loss = avg_neg_logpx_z + avg_KLD

            total_loss += loss
            total_neg_logpx_z += avg_neg_logpx_z
            total_kl += avg_KLD

            is_estimate = model.get_IS_estimate(x_seq, n_samples=n_is_samples)
            total_is_estimate += is_estimate.sum() / (x.shape[0] * seq_len)

            num_batches += 1

    return total_loss, total_neg_logpx_z, total_kl, total_is_estimate, num_batches
