# Neural Wave Machines: Learning Spatiotemporally Structured Representations with Locally Coupled Oscillatory Recurrent Neural Networks

### Visualizing and Measuring Waves on Rotating MNIST

This repository is built as a modification of the original public Topographic VAE library: (https://github.com/akandykeller/TopographicVAE)

## Getting Started
#### Install requirements with Anaconda:
`conda env create -f environment.yml`

#### Activate the conda environment
`conda activate tvae`

#### Install the tcornn package
Install the tvae package inside of your conda environment. This allows you to run experiments with the `tvae` command. At the root of the project directory run (using your environment's pip):
`pip3 install -e .`

If you need help finding your environment's pip, try `which python`, which should point you to a directory such as `.../anaconda3/envs/tvae/bin/` where it will be located.

#### (Optional) Setup Weights & Biases:
This repository uses Weight & Biases for experiment tracking. By deafult this is set to off. However, if you would like to use this (highly recommended!) functionality, all you have to do is set `'wandb_on': True` in the experiment config, and set your account's project and entity names in the `tvae/utils/logging.py` file.

For more information on making a Weight & Biases account see [(creating a weights and biases account)](https://app.wandb.ai/login?signup=true) and the associated [quickstart guide](https://docs.wandb.com/quickstart).

## Running an experiment
- `tvae --name 'tcornn_1d`
- `tvae --name 'tcornn_2d`