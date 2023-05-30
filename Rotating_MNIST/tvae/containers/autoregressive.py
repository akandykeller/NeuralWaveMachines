import torch

class Topographic_Autoregressive_NoConcat(torch.nn.Module):
    def __init__(self, encoder, decoder, rnn, grouper, ic_encoder=None, steps_ahead=1):
        super(Topographic_Autoregressive_NoConcat, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rnn = rnn
        self.grouper = grouper
        self.steps_ahead = steps_ahead
        self.ic_encoder = ic_encoder

    def forward(self, x_t, h_tm1, x_tp1=None):
        z_t, kl_z, log_q_z, log_p_z = self.encoder(x_t)
        half = int(z_t.shape[1] // 2)
        s = self.grouper(z_t[:, :half], z_t[:, half:])

        x_dec = x_tp1 if x_tp1 is not None and self.steps_ahead > 0 else x_t
        h_t = self.rnn(h_tm1, s)

        h_t_det = self.rnn.preprocess_for_decoder(h_t)
        probs_x, neg_log_px_z = self.decoder(h_t_det, x_dec)

        return s, None, h_t, probs_x, kl_z, neg_log_px_z

    def plot_forward_roll(self, h_init, t_fwd, dt=1):
        bsz, n_caps, cap_dim = h_init.shape
        recon = []
        for t in range(t_fwd):
            shifts = int(t * dt)
            shifts_p1 = shifts + 1
            blend = t * dt - shifts

            h_t = torch.roll(h_init, shifts=shifts, dims=-1).flatten(start_dim=1).to('cuda')
            h_tp1 = torch.roll(h_init, shifts=shifts_p1, dims=-1).flatten(start_dim=1).to('cuda')

            h_dec = (1.0 - blend) * h_t + (blend) * h_tp1

            probs_x = self.decoder.only_decode(h_dec)
            recon.append(probs_x)

        all_recon = torch.stack(recon, dim=1)
        return all_recon

    def init_hidden(self, x_seq):
        if self.ic_encoder is None:
            ic = torch.zeros((x_seq.shape[0], self.rnn.init_cond_size), device=x_seq.device)
            z_t, kl_z, log_q_z, log_p_z = self.encoder(x_seq[:, 0])
            half = int(z_t.shape[1] // 2)
            s = self.grouper(z_t[:, :half], z_t[:, half:])
            ic[:, :s.shape[1]] = s
        else:
            ## Reverse input order and pass to Init-Cond encoder-rnn
            out, (h_c, c_n) = self.ic_encoder(torch.flip(x_seq, (1,)).flatten(start_dim=2))
            # Take last output as initial state (shape: T, N, H)
            ic = out[:, -1].view(x_seq.shape[0], self.rnn.init_cond_size)
        return ic

    def plot_decoder_weights(self, wandb_on=True):
        self.decoder.plot_weights(name='Decoder Weights', wandb_on=wandb_on)

    def plot_encoder_weights(self, wandb_on=True):
        self.encoder.plot_weights(name='Encoder Weights', wandb_on=wandb_on)

    def forward_loglikelihood(self, x_t, h_tm1, x_tp1=None):
        z_t, kl_z, log_q_z, log_p_z = self.encoder(x_t)
        
        x_dec = x_tp1 if x_tp1 is not None and self.steps_ahead > 0 else x_t

        half = int(z_t.shape[1] // 2)
        s = self.grouper(z_t[:, :half], z_t[:, half:])

        x_dec = x_tp1 if x_tp1 is not None and self.steps_ahead > 0 else x_t
        h_t = self.rnn(h_tm1, s)

        h_t_det = self.rnn.preprocess_for_decoder(h_t)
        probs_x, neg_log_px_z = self.decoder(h_t_det, x_dec)

        nll = (-1 * neg_log_px_z.flatten(start_dim=1).sum(-1)
               + log_p_z.flatten(start_dim=1).sum(-1)
               - log_q_z.flatten(start_dim=1).sum(-1))
        return h_t, nll

    def get_IS_estimate(self, x, n_samples=100):
        log_likelihoods = []

        for n in range(n_samples):
            h_tm1 = self.init_hidden(x_seq=x)
            
            seq_kl = 0.0
            seq_neg_log_px_z = 0.0
            seq_lls = []
            for i in range(x.shape[1]):
                i_p1 = (i+1) % x.shape[1]
                h_t, nll_i = self.forward_loglikelihood(x[:, i], h_tm1, x[:, i_p1])
                seq_lls.append(nll_i)
                h_tm1 = h_t

            log_likelihoods.append(torch.cat(seq_lls, dim=0).unsqueeze(-1))  
        ll = torch.cat(log_likelihoods, dim=-1)
        is_estimate = torch.logsumexp(ll, -1)
        return is_estimate