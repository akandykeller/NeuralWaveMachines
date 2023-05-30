from torch import nn
import torch
import numpy as np
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class coRNNCell(nn.Module):
    def __init__(self, n_inp, spatial, dt, gamma, epsilon):
        super(coRNNCell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.spatial = spatial
        self.Wx = nn.Linear(n_inp, spatial * spatial).to(device)
        self.Wz = nn.Conv2d(1, 1, 3, padding=1, padding_mode='circular', bias=False).to(device)
        self.Wy = nn.Conv2d(1, 1, 3, padding=1, padding_mode='circular', bias=False).to(device)

    def forward(self,x,hy,hz):
        hz = hz + self.dt * (torch.tanh(self.Wx(x) 
                                         + self.Wz(hz.view(-1, 1, self.spatial, self.spatial)).flatten(start_dim=1)
                                         + self.Wy(hy.view(-1, 1, self.spatial, self.spatial)).flatten(start_dim=1))
                                   - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz

        return hy, hz

class coRNN(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon):
        super(coRNN, self).__init__()
        self.spatial = int(np.sqrt(n_hid))
        self.n_hid = int(self.spatial ** 2.0)
        self.cell = coRNNCell(n_inp,self.spatial,dt,gamma,epsilon)

    def forward(self, x, get_seq=False):
        ## initialize hidden states
        hy = Variable(torch.zeros(x.size(1),self.n_hid)).to(device)
        hz = Variable(torch.zeros(x.size(1),self.n_hid)).to(device)

        y_seq = []

        for t in range(x.size(0)):
            hy, hz = self.cell(x[t],hy,hz)
            if get_seq:
                y_seq.append(hy[0].view(1, self.spatial, self.spatial).detach().cpu())

        if get_seq:
            y_seq = torch.stack(y_seq, dim=0)
        
        return hy, y_seq



class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, dt, gamma, epsilon):
        super().__init__()
        self.spatial = int(np.sqrt(hidden_dim))
        self.hidden_dim = int(self.spatial ** 2.0)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx).to(device)
        self.rnn = coRNN(embedding_dim, self.hidden_dim,dt,gamma,epsilon).to(device)
        self.readout = nn.Linear(self.hidden_dim, output_dim).to(device)

    def forward(self, text, text_lengths, get_seq=False):
        embedded = self.embedding(text)
        hidden, seq = self.rnn(embedded, get_seq)
        return self.readout(hidden), seq
