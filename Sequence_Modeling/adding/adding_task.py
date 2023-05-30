from torch import nn, optim
import torch
import conv2d_model as model_tcornn2d
import model as model_cornn
import torch.nn.utils
import utils
import argparse
from tqdm import tqdm
from pathlib import Path
import os
from sys import exit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--run_name', type=str, default=None,
                    help='name of run for wandb')
parser.add_argument('--dataset', type=str, default='adding',
                    help='dataset name for wandb')
parser.add_argument('--model_type', type=str, default='cornn',
                    help='type of model, cornn, tcornn2d')
parser.add_argument('--n_hid', type=int, default=128,
                    help='hidden size of recurrent net')
parser.add_argument('--T', type=int, default=500,
                    help='length of sequences')
parser.add_argument('--max_steps', type=int, default=30000,
                    help='max learning steps')
parser.add_argument('--log_interval', type=int, default=100,
                    help='log interval')
parser.add_argument('--batch', type=int, default=50,
                    help='batch size')
parser.add_argument('--batch_test', type=int, default=1000,
                    help='size of test set')
parser.add_argument('--lr', type=float, default=2e-2,
                    help='learning rate')
parser.add_argument('--dt',type=float, default=1.6e-2,
                    help='step size <dt> of the coRNN')
parser.add_argument('--gamma',type=float, default=94.5,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon',type=float, default = 9.5,
                    help='z controle parameter <epsilon> of the coRNN')
parser.add_argument('--wandb', dest='wandb', default=False, action='store_true',
                    help='Use weights and biases to log videos of hidden state and training stats')         
args = parser.parse_args()

n_inp = 2
n_out = 1

model_select = {'cornn': model_cornn, 'tcornn2d': model_tcornn2d}

model = model_select[args.model_type].coRNN(n_inp, args.n_hid, n_out, args.dt, args.gamma, args.epsilon).to(device)

if args.wandb:
    import wandb
    wandb.init(name=args.run_name,
                project='PROJECT_NAME', 
                entity='ENTITY_NAME', 
                dir='WANDB_DIR',
                config=args)                
    wandb.watch(model)

def log(key, val):
    print(f"{key}: {val}")
    if args.wandb:
        wandb.log({key: val})

objective = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

fname = f'result/adding_test_log_{args.model_type}_h{args.n_hid}_T{args.T}.txt'

def test():
    model.eval()
    with torch.no_grad():
        data, label = utils.get_batch(args.T, args.batch_test)
        label = label.unsqueeze(1)
        out, _ = model(data.to(device))
        loss = objective(out, label.to(device))

    return loss.item()

solved = False
thresh = 0.05

test_mse = []
for i in tqdm(range(args.max_steps), desc=f"Adding_{args.model_type}_h{args.n_hid}, T{args.T}"):
    data, label = utils.get_batch(args.T,args.batch)
    label = label.unsqueeze(1)

    optimizer.zero_grad()
    out, _  = model(data.to(device))
    loss = objective(out, label.to(device))
    loss.backward()
    optimizer.step()

    if(i%100==0 and i!=0):
        log('Train Loss:', loss)

        mse_error = test()
        log('Test MSE:', mse_error)
        test_mse.append(mse_error)

        if solved == False and loss <= thresh:
            solved = True
            log('Solved Iter', i)
            exit()

        if i % 100 == 0:
            log('Train Loss:', loss)
            _, y_seq = model(data.to(device), get_seq=True)
            if args.wandb:
                utils.Plot_Vid(y_seq, fps=20)
            if torch.isnan(loss):
                exit()

        model.train()

        Path('result').mkdir(parents=True, exist_ok=True)
        f = open(fname, 'a')
        f.write('test mse: ' + str(round(test_mse[-1], 2)) + '\n')
        f.close()

        if torch.isnan(loss):
            if not solved:
                log('Solved Iter', args.max_steps)
            exit()

if not solved:
    log('Solved Iter', args.max_steps)