from torch import nn
import torch
from torch.autograd import Variable
import math

class coRNNCell(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, device):
        super(coRNNCell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.i2h = nn.Linear(n_inp + n_hid + n_hid, n_hid).to(device)

    def forward(self,x,hy,hz):
        hz = hz + self.dt * (torch.tanh(self.i2h(torch.cat((x, hz, hy),1)))
                                   - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz

        return hy, hz

class coRNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, dt, gamma, epsilon, device):
        super(coRNN, self).__init__()
        self.n_hid = n_hid
        sqrt = int(math.sqrt(n_hid))
        self.vis_h = max([int(int(n_hid / x) == (n_hid / x))*x for x in range(sqrt, 0, -1)])
        self.vis_w = self.n_hid // self.vis_h
        assert self.n_hid == self.vis_h * self.vis_w
        self.cell = coRNNCell(n_inp,n_hid,dt,gamma,epsilon, device)
        self.readout = nn.Linear(n_hid, n_out).to(device)
        self.device = device

    def forward(self, x, get_seq=False):
        ## initialize hidden states
        hy = Variable(torch.zeros(x.size(1),self.n_hid)).to(self.device)
        hz = Variable(torch.zeros(x.size(1),self.n_hid)).to(self.device)

        y_seq = []

        for t in range(x.size(0)):
            hy, hz = self.cell(x[t],hy,hz)
            if get_seq:
                y_seq.append(hy[0].view(1, self.vis_h, self.vis_w).detach().cpu())
        output = self.readout(hy)

        if get_seq:
            y_seq = torch.stack(y_seq, dim=0)

        return output, y_seq
