from torch import nn, optim
import torch
import uncoupled_model as model_uncoupled
import multiconv1d_model as model_tcornn1d
import conv2d_model as model_tcornn2d
import network as model_cornn
import torch.nn.utils
import utils
from pathlib import Path
import argparse
from tqdm import tqdm
import os
from sys import exit

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--run_name', type=str, default=None,
                    help='name of run for wandb')
parser.add_argument('--dataset', type=str, default='sMNIST',
                    help='')
parser.add_argument('--model_type', type=str, default='cornn',
                    help='type of model, cornn, tcornn1d, tcornn2d')
parser.add_argument('--n_hid', type=int, default=256,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, default=120,
                    help='max epochs')
parser.add_argument('--batch', type=int, default=120,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0021,
                    help='learning rate')
parser.add_argument('--dt', type=float, default=0.042, 
                    help='step size <dt> of the coRNN')
parser.add_argument('--gamma', type=float, default=2.7,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon', type=float, default=4.7,
                    help='z controle parameter <epsilon> of the coRNN')
parser.add_argument('--wandb', dest='wandb', default=False, action='store_true',
                    help='Use weights and biases to log videos of hidden state and training stats')

args = parser.parse_args()
print(args)

# torch.manual_seed(46159)

n_inp = 1
n_out = 10
bs_test = 1000

device='cuda'

model_select = {'cornn': model_cornn, 'tcornn1d': model_tcornn1d, 'tcornn2d': model_tcornn2d, 'uncoupled': model_uncoupled}

model = model_select[args.model_type].coRNN(n_inp, args.n_hid, n_out, args.dt, args.gamma, args.epsilon, device)
train_loader, valid_loader, test_loader = utils.get_data(args.batch,bs_test)

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

objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

fname = f'result/sMNIST_log_{args.model_type}_h{args.n_hid}.txt'

def test(data_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(data_loader, desc="Test")):
            images = images.reshape(bs_test, 1, 784)
            images = images.permute(2, 0, 1).to(device)
            labels = labels.to(device)

            output, _  = model(images)
            test_loss += objective(output, labels).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
    test_loss /= i+1
    accuracy = 100. * correct / len(data_loader.dataset)

    return accuracy.item()

for epoch in range(args.epochs):
    model.train()
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        images = images.reshape(args.batch, 1, 784)
        images = images.permute(2, 0, 1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output, _ = model(images, get_seq=False)
        loss = objective(output, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            log('Train Loss:', loss)
            _, y_seq = model(images, get_seq=True)
            if args.wandb:
                utils.Plot_Vid(y_seq)
            if torch.isnan(loss):
                exit()

    valid_acc = test(valid_loader)
    test_acc = test(test_loader)
    log('Valid Acc:', valid_acc)
    log('Test Acc:', test_acc)

    Path('result').mkdir(parents=True, exist_ok=True)
    f = open(fname, 'a')
    if (epoch == 0):
        f.write('## learning rate = ' + str(args.lr) + ', dt = ' + str(args.dt) + ', gamma = ' + str(args.gamma) + ', epsilon = ' + str(args.epsilon) + '\n')
    f.write('eval accuracy: ' + str(round(valid_acc, 2)) + '\n')
    f.close()

    if (epoch+1) % 100 == 0:
        args.lr /= 10.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

log('Final Test Acc:', test_acc)
f = open(fname, 'a')
f.write('final test accuracy: ' + str(round(test_acc, 2)) + '\n')
f.close()
