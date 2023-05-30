from torchtext.legacy import data
from torchtext.legacy import datasets
import torch
import wandb

def get_data(bs,embedding_size):
    text = data.Field(tokenize='spacy', include_lengths=True)
    label = data.LabelField(dtype=torch.float)
    train_data, test_data = datasets.IMDB.splits(text, label, 
                                                 # root='.../torchtext/data/'
                                                 )
    train_data, valid_data = train_data.split()

    max_vocab_size = 25_000
    text.build_vocab(train_data,
                     max_size=max_vocab_size,
                     vectors="glove.6B."+str(embedding_size)+"d",
                     unk_init=torch.Tensor.normal_,
                     # vectors_cache='.../torchtext/.vectors_cache/'
                     )
    label.build_vocab(train_data)
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                               batch_size=bs,sort=False)

    return train_iterator, valid_iterator, test_iterator, text

def zero_words_in_embedding(model, embedding_size, text, pad_idx):
    pretrained_embeddings = text.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = text.vocab.stoi[text.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(embedding_size)
    model.embedding.weight.data[pad_idx] = torch.zeros(embedding_size)

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
