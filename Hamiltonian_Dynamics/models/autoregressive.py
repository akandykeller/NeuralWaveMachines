# Copyright 2020 DeepMind Technologies Limited.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module for all autoregressive models."""
import functools
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import distrax
import haiku as hk
from jax import lax
import jax.numpy as jnp
import jax.random as jnr

import physics_inspired_models.metrics as metrics
import physics_inspired_models.models.base as base
import physics_inspired_models.models.networks as nets
import physics_inspired_models.utils as utils


class TeacherForcingAutoregressiveModel(base.SequenceModel):
  """A standard autoregressive model trained via teacher forcing."""

  def __init__(
      self,
      latent_system_dim: int,
      latent_system_net_type: str,
      latent_system_kwargs: Dict[str, Any],
      latent_dynamics_type: str,
      encoder_aggregation_type: Optional[str],
      decoder_de_aggregation_type: Optional[str],
      encoder_kwargs: Dict[str, Any],
      decoder_kwargs: Dict[str, Any],
      num_inference_steps: int,
      num_target_steps: int,
      name: Optional[str] = None,
      **kwargs
  ):

    # Remove any parameters from vae models
    encoder_kwargs = dict(**encoder_kwargs)
    encoder_kwargs["distribution_name"] = None

    if kwargs.get("has_latent_transform", False):
      raise ValueError("We do not support AR models with latent transform.")

    for k in ('dt', 'elbo_beta_delay', 'elbo_beta_final', 'geco_alpha', 'geco_kappa', 'latent_training_type', 'objective_type', 'training_data_split'):
      kwargs.pop(k, None)

    super().__init__(
        can_run_backwards=False,
        latent_system_dim=latent_system_dim,
        latent_system_net_type=latent_system_net_type,
        latent_system_kwargs=latent_system_kwargs,
        encoder_aggregation_type=encoder_aggregation_type,
        decoder_de_aggregation_type=decoder_de_aggregation_type,
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
        num_inference_steps=num_inference_steps,
        num_target_steps=num_target_steps,
        name=name,
        **kwargs
    )
    self.latent_dynamics_type = latent_dynamics_type

    # Arguments checks
    if self.latent_system_net_type != "mlp":
      raise ValueError("Currently we do not support non-mlp AR models.")

    def recurrence_function(sequence, initial_state=None):
      core = nets.make_flexible_recurrent_net(
          core_type=latent_dynamics_type,
          net_type=latent_system_net_type,
          output_dims=self.latent_system_dim,  # unused by cornn
          **self.latent_system_kwargs)
      initial_state = initial_state or core.initial_state(sequence.shape[1])
      core(sequence[0], initial_state)
      return hk.dynamic_unroll(core, sequence, initial_state)

    self.recurrence = hk.transform(recurrence_function)

  def process_inputs_for_encoder(self, x: jnp.ndarray) -> jnp.ndarray:
    return x

  def process_latents_for_dynamics(self, z: jnp.ndarray) -> jnp.ndarray:
    return z

  def process_latents_for_decoder(self, z: jnp.ndarray) -> jnp.ndarray:
    # if self.latent_dynamics_type == 'cornn':
      # return z[:,:,::2]
    return z

  @property
  def inferred_index(self) -> int:
    return self.num_inference_steps - 1

  @property
  def train_sequence_length(self) -> int:
    return self.num_target_steps

  def train_data_split(
      self,
      images: jnp.ndarray
  ) -> Tuple[jnp.ndarray, jnp.ndarray, Mapping[str, Any]]:
    images = images[:, :self.train_sequence_length]
    inference_data = images[:, :-1]
    target_data = images[:, 1:]
    return inference_data, target_data, dict(
        num_steps_forward=1,
        num_steps_backward=0,
        include_z0=False)

  def unroll_without_inputs(
      self,
      params: utils.Params,
      rng: jnp.ndarray,
      x_init: jnp.ndarray,
      h_init: jnp.ndarray,
      num_steps: int,
      is_training: bool
  ) -> Tuple[Tuple[distrax.Distribution, jnp.ndarray], Any]:
    if num_steps < 1:
      raise ValueError("`num_steps` must be at least 1.")

    def step_fn(carry, key):
      x_last, h_last = carry
      enc_key, dec_key = jnr.split(key)
      z_in_next = self.encoder.apply(params, enc_key, x_last,
                                     is_training=is_training)
      z_next, h_next = self.recurrence.apply(params, None, z_in_next[None],
                                             h_last)
      
      # Do we need this?
      z_next_dec = self.process_latents_for_decoder(z_next)

      p_x_next = self.decode_latents(params, dec_key, z_next_dec[0],
                                     is_training=is_training)
      return (p_x_next.mean(), h_next), (p_x_next, z_next[0])

    return lax.scan(
        step_fn,
        init=(x_init, h_init),
        xs=jnr.split(rng, num_steps)
    )

  def phase_unroll(
      self,
      z_full: jnp.ndarray,
      num_steps: int,
  ) -> Tuple[Tuple[distrax.Distribution, jnp.ndarray], Any]:
    if num_steps < 1:
      raise ValueError("`num_steps` must be at least 1.")
    
    z_init = utils.get_anal_init(z_full)

    z_t = [jnp.real(z_init)]
    z_anal_tp1 = z_init
    for t in range(num_steps):
      z_anal_tp1, z_real_tp1 = utils.phase_roll(z_anal_tp1)
      # z_real_tp1 = z_real_tp1.at[:, :z_real_tp1.shape[1]//2].set(z_t[-1][:, :z_real_tp1.shape[1]//2] + 0.125 * z_t[-1][:, z_real_tp1.shape[1]//2:])
      
      z_tp1 = z_t[-1].at[:, ::2].set(z_t[-1][:, ::2] + 0.125 * z_real_tp1[:, 1::2])
      z_t.append(z_tp1)

    z_t = jnp.stack(z_t, axis=0)
    return z_t
  

  def unroll_latent_dynamics(
      self,
      z: jnp.ndarray,
      params: utils.Params,
      key: jnp.ndarray,
      num_steps_forward: int,
      num_steps_backward: int,
      include_z0: bool,
      is_training: bool,
      phase_roll: bool = False,
      **kwargs: Any
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    init_key, unroll_key, dec_key = jnr.split(key, 3)

    if num_steps_backward != 0:
      raise ValueError("This model can not run backwards.")

    # Change 'z' time dimension to be first
    z = jnp.swapaxes(z, 0, 1)

    # Run recurrent model on inputs
    z_0, h_0 = self.recurrence.apply(params, init_key, z)    ## Pass in inferred starting state here

    if num_steps_forward == 1:
      z_t = z_0
    elif num_steps_forward > 1:
      # Do we need this?
      if not phase_roll:
        z_0_dec = self.process_latents_for_decoder(z_0)
        p_x_0 = self.decode_latents(params, dec_key, z_0_dec[-1], is_training=False)
        _, (_, z_t) = self.unroll_without_inputs(
            params=params,
            rng=unroll_key,
            x_init=p_x_0.mean(),
            h_init=h_0,
            num_steps=num_steps_forward-1,
            is_training=is_training
        )
        z_t = jnp.concatenate([z_0, z_t], axis=0)
      else:
        raise NotImplementedError
        # z_0_dec = self.process_latents_for_decoder(z_0)
        # z_t = self.phase_unroll(z_0[-1], num_steps_forward-1)
        # z_t = jnp.concatenate([z_0, z_t], axis=0)
    else:
      raise ValueError("num_steps_forward should be at least 1.")

    # Make time dimension second
    return jnp.swapaxes(z_t, 0, 1), dict()

  def _models_core(
      self,
      params: utils.Params,
      keys: jnp.ndarray,
      image_data: jnp.ndarray,
      is_training: bool,
      **unroll_kwargs: Any
  ) -> Tuple[distrax.Distribution, jnp.ndarray, jnp.ndarray]:
    enc_key, _, transform_key, unroll_key, dec_key, _ = keys

    # Calculate latent input representation
    inference_data = self.process_inputs_for_encoder(image_data)
    z_raw = self.encoder.apply(params, enc_key, inference_data,
                               is_training=is_training)

    # Apply latent transformation (should be identity)
    z0 = self.apply_latent_transform(params, transform_key, z_raw,
                                     is_training=is_training)
    z0 = self.process_latents_for_dynamics(z0)

    # Calculate latent output representation
    decoder_z, _ = self.unroll_latent_dynamics(
        z=z0,
        params=params,
        key=unroll_key,
        is_training=is_training,
        **unroll_kwargs
    )
    decoder_z = self.process_latents_for_decoder(decoder_z)

    # Compute p(x|z)
    p_x = self.decode_latents(params, dec_key, decoder_z,
                              is_training=is_training)
    return p_x, z0, decoder_z

  def training_objectives(
      self,
      params: hk.Params,
      state: hk.State,
      rng: jnp.ndarray,
      inputs: jnp.ndarray,
      step: jnp.ndarray,
      is_training: bool = True,
      use_mean_for_eval_stats: bool = True
  ) -> Tuple[jnp.ndarray, Sequence[Dict[str, jnp.ndarray]]]:
    """Computes the training objective and any supporting stats."""
    # Split all rng keys
    keys = jnr.split(rng, 6)

    # Process training data
    images = utils.extract_image(inputs)
    image_data, target_data, unroll_kwargs = self.train_data_split(images)

    p_x, _, decoder_z = self._models_core(
        params=params,
        keys=keys,
        image_data=image_data,
        is_training=is_training,
        **unroll_kwargs
    )

    # Compute training statistics
    stats = metrics.training_statistics(
        p_x=p_x,
        targets=target_data,
        rescale_by=self.rescale_by,
        p_x_learned_sigma=self.decoder_kwargs.get("learned_sigma", False)
    )

    # import ipdb; ipdb.set_trace()
    # lsk = self.latent_system_kwargs
    # spatial_shape = (-1, 1, lsk['spatial_dim'], lsk['spatial_dim'], lsk['n_state_channels'])
    # spatial_latents = decoder_z[0].reshape(spatial_shape)[:,:,:,:,0]
    # rgb = utils.normalize_int(utils.to_numpy(utils.rgb_phase(spatial_latents)))
    # import wandb
    # wandb_videos_w_mode = {'phase' + "_" + 'train' :
    #                         wandb.Video(rgb, fps=20, format='gif')
    #                       } 
    # wandb.log(wandb_videos_w_mode)



    # lsk = self.latent_system_kwargs
    # spatial_shape = (-1, lsk['spatial_dim'], lsk['spatial_dim'], lsk['n_state_channels'])
    # div_vel = utils.phase_divergence_loss(decoder_z, spatial_shape)
    # stats['div_loss'] = jnp.abs(div_vel).sum(axis=[1,2,3]) / (div_vel.shape[1] * div_vel.shape[2] * div_vel.shape[3])

    stats["loss"] = stats["neg_log_p_x"] # + stats['div_loss']
    # # The loss is just the negative log-likelihood (e.g. the L2 loss)
    # stats["loss"] = stats["neg_log_p_x"]

    if not is_training:
      # Optionally add the evaluation stats when not training
      # Add also the evaluation statistics
      # We need to be able to set `use_mean = False` for some of the tests
      stats.update(metrics.evaluation_only_statistics(
          reconstruct_func=functools.partial(
              self.reconstruct, use_mean=use_mean_for_eval_stats),
          params=params,
          inputs=inputs,
          rng=rng,
          rescale_by=self.rescale_by,
          can_run_backwards=self.can_run_backwards,
          train_sequence_length=self.train_sequence_length,
          reconstruction_skip=1,
          p_x_learned_sigma=self.decoder_kwargs.get("learned_sigma", False)
      ))

    return stats["loss"], (dict(), stats, dict())

  def reconstruct(
      self,
      params: utils.Params,
      inputs: jnp.ndarray,
      rng: jnp.ndarray,
      forward: bool,
      use_mean: bool = True,
      phase_roll: bool = False,
  ) -> distrax.Distribution:
    """Reconstructs the input sequence."""
    if not forward:
      raise ValueError("This model can not run backwards.")
    images = utils.extract_image(inputs)
    image_data = images[:, :self.num_inference_steps]

    p_x, z0, decoder_z =  self._models_core(
        params=params,
        keys=jnr.split(rng, 6),
        image_data=image_data,
        is_training=False,
        num_steps_forward=images.shape[1] - self.num_inference_steps,
        num_steps_backward=0,
        include_z0=False,
        phase_roll=phase_roll,
        )
    
    return p_x, decoder_z

  def gt_state_and_latents(
      self,
      params: hk.Params,
      rng: jnp.ndarray,
      inputs: Dict[str, jnp.ndarray],
      seq_length: int,
      is_training: bool = False,
      unroll_direction: str = "forward",
      **kwargs: Dict[str, Any]
  ) -> Tuple[jnp.ndarray, jnp.ndarray,
             Union[distrax.Distribution, jnp.ndarray]]:
    """Computes the ground state and matching latents."""
    assert unroll_direction == "forward"
    images = utils.extract_image(inputs)
    gt_state = utils.extract_gt_state(inputs)
    image_data = images[:, :self.num_inference_steps]
    gt_state = gt_state[:, 1:seq_length + 1]

    _, z_in, z_out = self._models_core(
        params=params,
        keys=jnr.split(rng, 6),
        image_data=image_data,
        is_training=False,
        num_steps_forward=images.shape[1] - self.num_inference_steps,
        num_steps_backward=0,
        include_z0=False,
    )

    return gt_state, z_out, z_in

  def _init_non_model_params_and_state(
      self,
      rng: jnp.ndarray
  ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    return dict(), dict()

  def _init_latent_system(
      self,
      rng: jnp.ndarray,
      z: jnp.ndarray,
      **kwargs: Any
  ) -> utils.Params:
    return self.recurrence.init(rng, z)



class InitCondAutoregressiveModel(TeacherForcingAutoregressiveModel):
  """A subclass of the TF-AR model which uses an RNN run backwards
      over the input sequence to predict the initial conditions
      of the latent system."""
  def __init__(
      self,
      latent_system_dim: int,
      latent_system_net_type: str,
      latent_system_kwargs: Dict[str, Any],
      latent_dynamics_type: str,
      encoder_aggregation_type: Optional[str],
      decoder_de_aggregation_type: Optional[str],
      encoder_kwargs: Dict[str, Any],
      decoder_kwargs: Dict[str, Any],
      num_ic_steps: int,
      num_inference_steps: int,
      num_target_steps: int,
      name: Optional[str] = None,
      **kwargs
  ):
    assert num_ic_steps <= num_target_steps
    assert num_ic_steps <= num_inference_steps
    self.num_ic_steps = num_ic_steps

    # Initalize RNN and Linear Layer for predicting IC (as hk.transforms)

    super().__init__(
        latent_system_dim=latent_system_dim,
        latent_system_net_type=latent_system_net_type,
        latent_system_kwargs=latent_system_kwargs,
        latent_dynamics_type=latent_dynamics_type,
        encoder_aggregation_type=encoder_aggregation_type,
        decoder_de_aggregation_type=decoder_de_aggregation_type,
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
        num_inference_steps=num_inference_steps,
        num_target_steps=num_target_steps,
        name=name,
        **kwargs
    )

    self.full_latent_system_dim = int(# self.latent_system_kwargs['spatial_dim'] ** 2
                                      self.latent_system_kwargs['spatial_height'] * self.latent_system_kwargs['spatial_width']
                                      * self.latent_system_kwargs['n_components']
                                      * self.latent_system_kwargs['n_state_channels'])
                      
    def recurrence_function(sequence, initial_state):
      core = nets.make_flexible_recurrent_net(
          core_type=latent_dynamics_type,
          net_type=latent_system_net_type,
          output_dims=self.latent_system_dim, # unused by cornn
          **self.latent_system_kwargs)
      core(sequence[0], initial_state)
      return hk.dynamic_unroll(core, sequence, initial_state)

    self.recurrence = hk.transform(recurrence_function)

    def ic_recurrence_function(sequence, initial_state=None):
      core = nets.make_flexible_recurrent_net(
          core_type='GRU',
          net_type='mlp',
          output_dims=self.full_latent_system_dim,
          net_kwargs={'num_layers': 1, 
                      'num_units': self.full_latent_system_dim,
                      'activation':lambda x: x},
          name='IC_Net')
      initial_state = initial_state or core.initial_state(sequence.shape[1])
      core(sequence[0], initial_state)
      return hk.dynamic_unroll(core, sequence, initial_state)

    self.ic_net = hk.transform(ic_recurrence_function)

  def unroll_latent_dynamics(
      self,
      z: jnp.ndarray,
      params: utils.Params,
      key: jnp.ndarray,
      num_steps_forward: int,
      num_steps_backward: int,
      include_z0: bool,
      is_training: bool,
      phase_roll: bool = False,
      **kwargs: Any
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    init_key, ic_key, unroll_key, dec_key = jnr.split(key, 4)

    if num_steps_backward != 0:
      raise ValueError("This model can not run backwards.")

    # Change 'z' time dimension to be first
    z = jnp.swapaxes(z, 0, 1)

    # Compute h_ic by running RNN backwards on z
    z_rev_ic = z[:self.num_ic_steps][::-1]
    _, h_ic = self.ic_net.apply(params, ic_key, z_rev_ic)

    # h_ic = (jnp.zeros_like(h_ic[0]),) ##### Ablation testing for no-ic encoder

    # Run recurrent model on inputs
    z_0, h_0 = self.recurrence.apply(params, init_key, z, h_ic)    ## Pass in inferred starting state here

    if num_steps_forward == 1:
      z_t = z_0
    elif num_steps_forward > 1:
      # Do we need this?
      z_0_dec = self.process_latents_for_decoder(z_0)
      p_x_0 = self.decode_latents(params, dec_key, z_0_dec[-1], is_training=False)
      _, (_, z_t) = self.unroll_without_inputs(
          params=params,
          rng=unroll_key,
          x_init=p_x_0.mean(),
          h_init=h_0,
          num_steps=num_steps_forward-1,
          is_training=is_training
      )
      z_t = jnp.concatenate([z_0, z_t], axis=0)

      if phase_roll:
        z_t = self.phase_unroll(z_t, z_t.shape[0])
        # z_t = jnp.concatenate([z_0, z_t], axis=0)
    else:
      raise ValueError("num_steps_forward should be at least 1.")

    # Make time dimension second
    return jnp.swapaxes(z_t, 0, 1), dict()

  def _init_latent_system(
      self,
      rng: jnp.ndarray,
      z: jnp.ndarray,
      **kwargs: Any
  ) -> utils.Params:
    cornn_key, ic_key = jnr.split(rng, 2)
    cornn_params = self.recurrence.init(cornn_key, z, jnp.zeros((z.shape[1], self.full_latent_system_dim)))
    ic_params = self.ic_net.init(ic_key, z)
    params = hk.data_structures.merge(cornn_params, ic_params)
    return params

  ### Re-use the encoder from the main model, and just run RNN backwards on that
  ### To compute the ICs... Otherwise need to re-implement the _models_core with the
  ### new encoder
  # def _models_core(
  #     self,
  #     params: utils.Params,
  #     keys: jnp.ndarray,
  #     image_data: jnp.ndarray,
  #     is_training: bool,
  #     **unroll_kwargs: Any
  # ) -> Tuple[distrax.Distribution, jnp.ndarray, jnp.ndarray]:
  #   enc_key, _, transform_key, unroll_key, dec_key, _ = keys

  #   # Calculate latent input representation
  #   inference_data = self.process_inputs_for_encoder(image_data)
  #   z_raw = self.encoder.apply(params, enc_key, inference_data,
  #                              is_training=is_training)

  #   # Apply latent transformation (should be identity)
  #   z0 = self.apply_latent_transform(params, transform_key, z_raw,
  #                                    is_training=is_training)
  #   z0 = self.process_latents_for_dynamics(z0)

  #   # Calculate latent output representation
  #   decoder_z, _ = self.unroll_latent_dynamics(
  #       z=z0,
  #       params=params,
  #       key=unroll_key,
  #       is_training=is_training,
  #       **unroll_kwargs
  #   )
  #   decoder_z = self.process_latents_for_decoder(decoder_z)

  #   # Compute p(x|z)
  #   p_x = self.decode_latents(params, dec_key, decoder_z,
  #                             is_training=is_training)
  #   return p_x, z0, decoder_z