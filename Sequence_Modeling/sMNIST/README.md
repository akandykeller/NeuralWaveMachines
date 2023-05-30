# Sequential MNIST
## Usage

```
python sMNIST_task.py [args]
```

Options:
- model_type: sets the model class ('cornn', 'uncoupled', 'tcornn1d', 'tcornn2d')
- wandb: Boolean -- use weights and biases logging to store videos of hidden state
- nhid : hidden size of recurrent net
- epochs : max epochs
- batch : batch size
- lr : learning rate
- dt : step size parameter dt of the coRNN
- gamma : y controle parameter gamma of the coRNN
- epsilon : z controle parameter epsilon of the coRNN