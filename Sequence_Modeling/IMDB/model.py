import torch
import torch.nn as nn
from torch.autograd import Variable
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class coRNN(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon):
        super(coRNN, self).__init__()
        self.n_hid = n_hid
        sqrt = int(math.sqrt(n_hid))
        self.vis_h = max([int(int(n_hid / x) == (n_hid / x))*x for x in range(sqrt, 0, -1)])
        self.vis_w = self.n_hid // self.vis_h
        assert self.n_hid == self.vis_h * self.vis_w
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.i2h = nn.Linear(n_inp, n_hid).to(device)
        self.h2h = nn.Linear(n_hid+n_hid, n_hid, bias=False).to(device)

    def forward(self, x, get_seq=False):
        hy = Variable(torch.zeros(x.size(1),self.n_hid)).to(device)
        hz = Variable(torch.zeros(x.size(1),self.n_hid)).to(device)

        y_seq = []

        inputs = self.i2h(x)
        for t in range(x.size(0)):
            hz = hz + self.dt * (torch.tanh(self.h2h(torch.cat((hz,hy),dim=1)) + inputs[t])
                                          - self.gamma * hy - self.epsilon * hz)
            hy = hy + self.dt * hz

            if get_seq:
                y_seq.append(hy[0].view(1, self.vis_h, self.vis_w).detach().cpu())

        if get_seq:
            y_seq = torch.stack(y_seq, dim=0)

        return hy, y_seq

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, dt, gamma, epsilon):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx).to(device)
        self.rnn = coRNN(embedding_dim, hidden_dim,dt,gamma,epsilon).to(device)
        self.readout = nn.Linear(hidden_dim, output_dim).to(device)

    def forward(self, text, text_lengths, get_seq=False):
        embedded = self.embedding(text)
        hidden, seq = self.rnn(embedded, get_seq)
        return self.readout(hidden), seq
