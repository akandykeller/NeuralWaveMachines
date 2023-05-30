from torch import nn, optim
import torch
import uncoupled_model as model_uncoupled
import multiconv1d_model as model_tcornn1d
import conv2d_model as model_tcornn2d
import model as model_cornn
import argparse
import utils
from tqdm import tqdm
import os
from sys import exit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--run_name', type=str, default=None,
                    help='name of run for wandb')
parser.add_argument('--model_type', type=str, default='cornn',
                    help='type of model, cornn, tcornn1d, tcornn2d')
parser.add_argument('--dataset', type=str, default='IMDB',
                    help='')
parser.add_argument('--n_hid', type=int, default=128,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, default=100,
                    help='max epochs')
parser.add_argument('--batch', type=int, default=64,
                    help='batch size')
parser.add_argument('--embedding', type=int, default=100,
                    help='embedding size for the dictonary')
parser.add_argument('--lr', type=float, default=6e-4,
                    help='learning rate')
parser.add_argument('--dt',type=float, default=5.4e-2,  # sigmoid(-2.86) = 5.4e-2,
                    help='step size <dt> of the coRNN')
parser.add_argument('--gamma',type=float, default=4.9,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon',type=float, default = 4.8,
                    help='z controle parameter <epsilon> of the coRNN')
parser.add_argument('--wandb', dest='wandb', default=False, action='store_true',
                    help='Use weights and biases to log videos of hidden state and training stats')         

args = parser.parse_args()
print(args)

device='cuda'

## set up data iterators and dictonary:
train_iterator, valid_iterator, test_iterator, text_field = utils.get_data(args.batch,args.embedding)

n_inp = len(text_field.vocab)
n_out = 1
pad_idx = text_field.vocab.stoi[text_field.pad_token]

model_select = {'cornn': model_cornn, 'tcornn1d': model_tcornn1d, 'tcornn2d': model_tcornn2d, 'uncoupled': model_uncoupled}

model =  model_select[args.model_type].RNNModel(n_inp,args.embedding,args.n_hid,n_out,
                       pad_idx, args.dt, args.gamma, args.epsilon).to(device)

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

## zero embedding for <unk_token> and <padding_token>:
utils.zero_words_in_embedding(model,args.embedding,text_field,pad_idx)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.BCEWithLogitsLoss()

fname = f'result/IMDB_log_{args.model_type}_h{args.n_hid}.txt'

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def evaluate(data_iterator):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_iterator, desc='Eval'):
            text, text_lengths = batch.text
            predictions, _ = model(text.to(device), text_lengths.to(device))
            predictions = predictions.squeeze(1)
            loss = criterion(predictions, batch.label.to(device))
            acc = binary_accuracy(predictions, batch.label.to(device))
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(data_iterator), epoch_acc / len(data_iterator)

def train():
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    i = 0
    for batch in tqdm(train_iterator, desc='Train'):
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions, _ = model(text.to(device), text_lengths.to(device))
        predictions = predictions.squeeze(1)
        loss = criterion(predictions, batch.label.to(device))
        acc = binary_accuracy(predictions, batch.label.to(device))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if i % 100 == 0:
            log('Train Loss:', loss)
            _, y_seq = model(text.to(device), text_lengths.to(device), get_seq=True)
            if args.wandb:
                utils.Plot_Vid(y_seq)
            if torch.isnan(loss):
                exit()

        i += 1
    return epoch_loss / len(train_iterator), epoch_acc / len(train_iterator)

if __name__ == "__main__":
    for epoch in range(args.epochs):
        log("Epoch", epoch)
        train_loss, train_acc = train()
        log("Train Loss", train_loss)
        log("Train Acc", train_acc)
        eval_loss, eval_acc = evaluate(valid_iterator)
        log("Eval Loss", eval_loss)
        log("Eval Acc", eval_acc)
        test_loss, test_acc = evaluate(test_iterator)
        log("Test Loss", test_loss)
        log("Test Acc", test_acc)