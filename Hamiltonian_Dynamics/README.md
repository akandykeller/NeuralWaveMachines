# Implementation of multiple physics inspired models for modelling dynamics

This directory contains a modification of [the original repository for the papers](https://github.com/deepmind/deepmind-research/tree/master/physics_inspired_models) 'SyMetric: Measuring the Quality of Learnt Hamiltonian
Dynamics Inferred from Vision' and 'Which priors matter? Benchmarking models for learning latent dynamics'.

## Installation

All package requirements are listed in `requirements.txt`.
You will still need to download and setup the datasets from the
[DeepMind Hamiltonian Dynamics Suite] manually.

```shell
git clone git@github.com:deepmind/deepmind-research.git
pip install -r ./deepmind_research/physics_inspired_models/requirements.txt
pip install ./deepmind_research/physics_inspired_models
pip install --upgrade "jax[XXX]"
```

where `XXX` is the correct type of accelerator that you have on your machine.
Note that if you are using a GPU you might need `XXX` to also include the
correct version of CUDA and cuDNN installed on your machine.
For more details please read [here](https://github.com/google/jax#installation).

## Usage

The file `jaxline_configs.py` contains all the configurations specifications for
the experiments in the two papers. To run an experiment, in addition to passing
the location of the configs file, you must provide extra arguments in the
following manner:

`${name_of_configuration},${index_in_sweep},${dataset_name}`

For example to run the second hyper-parameter configuration of the improved
Hamiltonian Generative Network (HGN++) on the mass-spring dataset you should
run in the command line (assuming that you are in the folder of the project):

```shell
python3 jaxline_train.py \
  --config="jaxline_configs.py:sym_metric_hgn_plus_plus_sweep,1,toy_physics/mass_spring" \
  --jaxline_mode="train" \
  --logtostderr
```


## References

**SyMetric: Measuring the Quality of Learnt Hamiltonian Dynamics Inferred from Vision**

Irina Higgins, Peter Wirnsberger, Andrew Jaegle, Aleksandar Botev

URL: https://openreview.net/forum?id=9Qu0U9Fj7IP


**Which priors matter? Benchmarking models for learning latent dynamics**

Aleksandar Botev, Drew Jaegle, Peter Wirnsberger, Daniel Hennes and Irina
Higgins

URL: https://openreview.net/forum?id=qBl8hnwR0px


[DeepMind Hamiltonian Dynamics Suite]: https://github.com/deepmind/dm_hamiltonian_dynamics_suite
