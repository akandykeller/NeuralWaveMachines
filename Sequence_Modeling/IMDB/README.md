# IMDB 

## Overview
The IMDB data set is a collection of written movie reviews. The aim of this binary sentiment classification task is to decide whether a movie review is positive or negative.
## Data
The data set consists of 50k movie reviews. 25k reviews are used for training 
(with 7.5k of them are used for evaluating) and 25k reviews are used for testing.
## Usage

```
python IMDB_task.py [args]
```

Options:
- model_type: sets the model class ('cornn', 'uncoupled', 'tcornn1d', 'tcornn2d')
- wandb: Boolean -- use weights and biases logging to store videos of hidden state
- nhid : hidden size of recurrent net
- epochs : max epochs
- batch : batch size
- embedding : embedding size for the dictonary
- lr : learning rate
- dt : step size parameter of the coRNN
- gamma : y controle parameter of the coRNN
- epsilon : z controle parameter of the coRNN
