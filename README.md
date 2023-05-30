# Neural Wave Machines: Learning Spatiotemporally Structured Representations with Locally Coupled Oscillatory Recurrent Neural Networks

<img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/rotating_mnist_full.gif" width="800" height="200" /> 

Official implementation of the paper: "Neural Wave Machines: Learning Spatiotemporally Structured Representations with Locally Coupled Oscillatory Recurrent Neural Networks" accepted at ICML 2023.

This repository contains all code necessary to reproduce the experiments in the paper and additionally includes video visualizations of the spatiotemporal dynamics for each dataset in the README below. 

## Repository Organization
This repository is orginzed into three core directories: 
- Rotating_MNIST, containing a modification of the original [Topographic VAE library](https://github.com/akandykeller/TopographicVAE) necessary to reproduce the results in Figures 1, 3, 4, & 5. 
- Hamiltonian_Dynamics, containing a modification of [the original Hamiltonian Neural Networks code](https://github.com/deepmind/deepmind-research/tree/master/physics_inspired_models) necessary to reproduce the results in Table 1 and Figure 2 pertaining to modeling simple physical dynamics.
- Sequence_Modeling, containing a modification of [the original Coupled Oscillatory Recurrent Neural Network code](https://github.com/tk-rusch/coRNN) necessary to reproduce the sequence modeling results on sequential MNIST, permuted sequential MNIST, IMDB sentiment classification, and the long sequence addition task shown in Tables 2 & 4.

Since the code is built as a modification of each of these three directories individually, we recommend separate environments and installation for each following their respective guidelines. 

## Hidden State Visualizations 
Below we show the hidden state and corresponding instantaneous phase for a variety of Neural Wave Machines and wave-free baselines on the datasets used in this paper.

#### 2D NWM on Rotating MNIST

Before Training: (left hidden state, right phase)

<img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_mnist/before_training_rot_mnist.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_mnist/before_training_rot_mnist_phase.gif" width="100" height="100" />

After Training: (left hidden state, right phase)

<img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_mnist/wave_s4main_pos.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_mnist/wave_s4main_phase.gif" width="100" height="100" />

Hidden state of same model but on different data samples:

<img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_mnist/wave_s1_pos.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_mnist/wave_s1_phase.gif" width="100" height="100" /> 
<img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_mnist/wave_s2_pos.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_mnist/wave_s2_phase.gif" width="100" height="100" />
<img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_mnist/wave_s3_pos.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_mnist/wave_s3_phase.gif" width="100" height="100" />
<img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_mnist/wave_s5_pos.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_mnist/wave_s5_phase.gif" width="100" height="100" />

Hidden state of 2D NWM with different random initalizations:

<img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_mnist/randinit2_waves_pos.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_mnist/randinit2_waves_phase.gif" width="100" height="100" /> 
<img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_mnist/randinit3_waves_pos.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_mnist/randinit3_waves_phase.gif" width="100" height="100" />
<img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_mnist/randinit4_waves_pos.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_mnist/randinit4_waves_phase.gif" width="100" height="100" />

#### 2D NWM on Spring Task 
(Ground Truth, Forward Extrapolated Reconstruction, Hidden State, Phase)

<img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_spring/gt.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_spring/recon.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_spring/pos.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_spring/phase.gif" width="100" height="100" /> 

#### (Baseline) Globally Coupled coRNN on Spring Task 
(Ground Truth, Forward Extrapolated Reconstruction, Hidden State, Phase)

<img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/coRNN_Spring/gt.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/coRNN_Spring/recon.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/coRNN_Spring/pos.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/coRNN_Spring/phase.gif" width="100" height="100" /> 


#### 2D NWM on Pendulum Task
<img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_pendulum/gt.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_pendulum/recon.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_pendulum/pos.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_pendulum/phase.gif" width="100" height="100" /> 

#### (Baselinne) Globally Coupled coRNN on Pendulum Task
<img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/coRNN_pendulum/gt.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/coRNN_pendulum/recon.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/coRNN_pendulum/pos.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/coRNN_pendulum/phase.gif" width="100" height="100" /> 

#### 2D NWM on 2-Body Task
<img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_2body/gt.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_2body/recon.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_2body/pos.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/NWM_2d_2body/phase.gif" width="100" height="100" /> 

#### (Baseline) Globally Coupled coRNN on 2-Body Task
<img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/coRNN_2body/gt.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/coRNN_2body/recon.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/coRNN_2body/pos.gif" width="100" height="100" /> <img src="https://github.com/akandykeller/NeuralWaveMachines/blob/master/figures/coRNN_2body/phase.gif" width="100" height="100" /> 

## Acknowledgements
The Robert Bosch GmbH is acknowledged for financial support.
