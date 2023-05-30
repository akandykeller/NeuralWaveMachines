# Adding task

## Overview
The adding task is used to test the ability of RNNs to learn very long-term dependencies. The input is a two-dimensional sequence of length, with first dimension consisting of random numbers drawn uniformly and with two non-zero entries in the second dimension, both set to $1$ and chosen at random, but one each in both halves of the sequence and the output is the sum of two numbers of the first dimension at the positions which are indicated by the two 1 entries in the second dimension.
## Data
As this task is not based on a given data set, the training and test data is artificially generated.
## Usage

```
python adding_task.py [args]
```

Options:
- model_type: sets the model class ('cornn', 'tcornn2d')
- wandb: Boolean -- use weights and biases logging to store videos of hidden state
- nhid : hidden size of recurrent net
- T : length of sequences
- max_steps : max learning steps
- log_interval : log interval
- batch : batch size
- batch_test : size of test set
- lr : learning rate
- dt : step size parameter dt of the coRNN
- gamma : y controle parameter gamma of the coRNN
- epsilon : z controle parameter epsilon of the coRNN
