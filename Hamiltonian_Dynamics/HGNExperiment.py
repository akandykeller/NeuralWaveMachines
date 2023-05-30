"""The training script for the HGN models."""
from absl import logging
from dm_hamiltonian_dynamics_suite import load_datasets
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import experiment
import numpy as np
import optax

import eval_metric
from physics_inspired_models import utils
from physics_inspired_models.models import common

AutoregressiveModel = common.autoregressive.TeacherForcingAutoregressiveModel

class HGNExperiment(experiment.AbstractExperiment):
  """HGN experiment."""
  CHECKPOINT_ATTRS = {
      "_params": "params",
      "_state": "state",
      "_opt_state": "opt_state",
  }
  NON_BROADCAST_CHECKPOINT_ATTRS = {
      "_python_step": "python_step"
  }

  def __init__(self, mode, init_rng, config):
    super().__init__(mode=mode)
    self.mode = mode
    self.init_rng = init_rng
    self.config = config

    # Checkpointed experiment state.
    self._python_step = None
    self._params = None
    self._state = None
    self._opt_state = None

    # Input pipelines.
    self._train_input = None
    self._step_fn = None
    self._burnin_fn = None
    self._eval_input = None
    self._eval_batch = None
    self._eval_input_metric = None
    self._eval_input_vpt = None
    self._compute_gt_state_and_latents = None
    self._get_reconstructions = None
    self._get_samples = None

    # Construct the model
    model_kwargs = dict(**self.config.model_kwargs)
    self.model = common.construct_model(**model_kwargs)
    # Construct the optimizer
    optimizer_ctor = getattr(optax, self.config.optimizer.name)
    self.optimizer = optimizer_ctor(**self.config.optimizer.kwargs)
    self.model_init = jax.pmap(self.model.init)
    self.opt_init = jax.pmap(self.optimizer.init)
    logging.info("Number of hosts: %d/%d",
                 jax.process_index(), jax.process_count())
    logging.info("Number of local devices: %d/%d", jax.local_device_count(),
                 jax.device_count())

  def _process_stats(self, stats, axis_name=None):
    keys_to_remove = list()
    for key in stats.keys():
      for dropped_keys in self.config.drop_stats_containing:
        if dropped_keys in key:
          keys_to_remove.append(key)
          break
    for key in keys_to_remove:
      stats.pop(key)
    # Take average statistics
    stats = jax.tree_map(utils.mean_if_not_scalar, stats)
    stats = utils.filter_only_scalar_stats(stats)
    if axis_name is not None:
      stats = utils.pmean_if_pmap(stats, axis_name="i")
    return stats

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #
  def step(self, global_step, rng, **unused_args):
    """See base class."""
    if self._train_input is None:
      self._initialize_train()

    # Do a small burnin to accumulate any persistent network state
    if self._python_step == 0 and self._state:
      for _ in range(self.config.training.burnin_steps):
        rng, key = utils.p_split(rng, 2)
        batch = next(self._train_input)
        self._state = self._burnin_fn(self._params, self._state, key, batch)
      self._state = jax.tree_map(
          lambda x: x / self.config.training.burnin_steps, self._state)

    batch = next(self._train_input)
    self._params, self._state, self._opt_state, stats = self._step_fn(
        self._params, self._state, self._opt_state, rng, batch, global_step)
    self._python_step += 1

    stats = utils.get_first(stats)
    logging.info("global_step: %d, %s", self._python_step,
                 jax.tree_map(float, stats))
    return stats

  def _initialize_train(self):
    self._train_input = utils.py_prefetch(
        load_datasets.dataset_as_iter(self._build_train_input))
    self._burnin_fn = jax.pmap(
        self._jax_burnin_fn, axis_name="i")# , donate_argnums=list(range(1, 4)))
    self._step_fn = jax.pmap(
        self._jax_train_step_fn, axis_name="i")#, donate_argnums=list(range(5)))

    if self._params is not None:
      logging.info("Not running initialization - loaded from checkpoint.")
      assert self._opt_state is not None
      return

    logging.info("Initializing parameters - NOT loading from checkpoint.")

    # Use the same rng on all devices, so that the initialization is identical
    init_rng = utils.bcast_local_devices(self.init_rng)

    # Initialize the parameters and the optimizer
    batch = next(self._train_input)
    self._params, self._state = self.model_init(init_rng, batch)
    self._python_step = 0
    self._opt_state = self.opt_init(self._params)

  def _build_train_input(self):
    batch_size = self.config.training.batch_size
    return load_datasets.load_dataset(
        path=self.config.dataset_folder,
        tfrecord_prefix="train",
        sub_sample_length=self.model.train_sequence_length,
        per_device_batch_size=batch_size,
        num_epochs=self.config.training.num_epochs,
        drop_remainder=True,
        multi_device=True,
        shuffle=True,
        shuffle_buffer=100 * batch_size,
        cache=False,
        keys_to_preserve=["image"],
    )

  def _jax_train_step_fn(self, params, state, opt_state, rng_key, batch, step):
    # The loss and the stats are averaged over the batch
    def loss_func(*args):
      outs = self.model.training_objectives(*args, is_training=True)
      # Average everything over the batch
      return jax.tree_map(utils.mean_if_not_scalar, outs)

    # Compute gradients
    grad_fn = jax.grad(loss_func, has_aux=True)
    grads, (state, stats, _) = grad_fn(params, state, rng_key, batch, step)
    # Average everything over the devices (e.g. average and sync)
    grads, state = utils.pmean_if_pmap((grads, state), axis_name="i")
    # Apply updates
    updates, opt_state = self.optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    # Weight Norm
    # pm = hk.data_structures.to_mutable_dict(params)

    # wc = pm['CORNN_0/~/CouplingNet']['w']
    # wd = pm['CORNN_0/~/DampingCouplingNet']['w']
    # wv = pm['CORNN_0/~/ForcingCouplingNet']['w']

    # wc = wc / wc.sum(axis=(0,1), keepdims=True)
    # wd = wd / wd.sum(axis=(0,1), keepdims=True)
    # wv = wv / wv.sum(axis=(0,1), keepdims=True)

    # w = pm['CoupledOscillator/~/CouplingNet/~/conv_2d_0']['w']
    # pm['CoupledOscillator/~/CouplingNet/~/conv_2d_0']['w'] = w / w * 1.0/9.0 #/ (jnp.linalg.norm(w) ** 2.0)
    # pm['CoupledOscillator/~/CouplingNet/~/conv_2d_0']['b'] = pm['CoupledOscillator/~/CouplingNet/~/conv_2d_0']['b'] * 0.0 
    # params = hk.data_structures.to_immutable_dict(pm)

    return params, state, opt_state, self._process_stats(stats, axis_name="i")

  def _jax_burnin_fn(self, params, state, rng_key, batch):
    _, (new_state, _, _) = self.model.training_objectives(
        params, state, rng_key, batch, jnp.zeros([]), is_training=True)
    new_state = jax.tree_map(utils.mean_if_not_scalar, new_state)
    new_state = utils.pmean_if_pmap(new_state, axis_name="i")
    new_state = hk.data_structures.to_mutable_dict(new_state)
    new_state = hk.data_structures.to_immutable_dict(new_state)
    return jax.tree_multimap(jnp.add, new_state, state)

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #
  def evaluate(self, global_step, rng, writer):
    """See base class."""
    logging.info("Starting evaluation.")
    if self.mode == "eval" or self.mode == "train_eval_multithreaded":
      if self._eval_input is None:
        self._initialize_eval()
        self._initialize_eval_vpt()
      key1, _ = utils.p_split(rng, 2)
      stats = utils.to_numpy(self._eval_epoch(global_step, key1))
      stats.update(utils.to_numpy(self._eval_epoch_vpt(global_step, rng, writer)))
    elif self.mode == "eval_metric":
      if self._eval_input_metric is None:
        self._initialize_eval_metric()
      stats = utils.to_numpy(self._eval_epoch_metric(global_step, rng))
    else:
      raise NotImplementedError()
    logging.info("Finished evaluation.")
    return stats

  def _eval_epoch(self, step, rng):
    """Evaluates an epoch."""
    accumulator = utils.MultiBatchAccumulator()
    for batch in self._eval_input():
      rng, key = utils.p_split(rng, 2)
      stats, num_samples = utils.get_first(
          self._eval_batch(self._params, self._state, key, batch, step)
      )
      accumulator.add(stats, num_samples)
    return accumulator.value()

  def _eval_epoch_metric(self, step, rng):
    """Evaluates an epoch."""
    # To prevent from calculating SyMetric early on in training where a large
    # polynomial expansion is likely to be required and the score is likely
    # to be bad anyway, we only compute using a single batch to save compute
    if step[0] > self.config.evaluation_metric.calculate_fully_after_steps:
      batch_n = self.config.evaluation_metric.batch_n
    else:
      batch_n = 1
    logging.info("Step: %d, batch_n: %d", step[0], batch_n)

    accumulator = utils.MultiBatchAccumulator()
    for _ in range(self.config.evaluation_metric.batch_n):
      batch = next(self._eval_input_metric)
      rng, key = utils.p_split(rng, 2)
      stats = self._eval_batch_metric(
          self._params, key, batch,
          eval_seq_len=self.config.evaluation_metric.num_eval_metric_steps,
          )
      accumulator.add(stats, 1)
    stats = utils.flatten_dict(accumulator.value())
    max_keys = ("sym", "SyMetric")
    for k, v in utils.flatten_dict(accumulator.max()).items():
      if any(m in k for m in max_keys):
        stats[k + "_max"] = v

    min_keys = ("sym", "SyMetric")
    for k, v in utils.flatten_dict(accumulator.min()).items():
      if any(m in k for m in min_keys):
        stats[k + "_min"] = v

    sum_keys = ("sym", "SyMetric")
    for k, v in utils.flatten_dict(accumulator.sum()).items():
      if any(m in k for m in sum_keys):
        stats[k + "_sum"] = v
    return stats

  def _eval_epoch_vpt(self, step, rng, writer):
    """Evaluates an epoch."""
    accumulator = utils.MultiBatchAccumulator()
    for _ in range(self.config.evaluation_vpt.batch_n):
      batch = next(self._eval_input_vpt)
      rng, key = utils.p_split(rng, 2)
      stats, movies = self._eval_batch_vpt(self._params, self._state, key, batch)
      accumulator.add(stats, 1)
    stats = utils.flatten_dict(accumulator.value())
    writer.write_videos(utils.get_first(step), utils.flatten_dict(movies), fps=20)
    return stats

  def _reconstruct_and_align(self, rng_key, full_trajectory, prefix, suffix):
    if hasattr(self.model, "training_data_split"):
      if self.model.training_data_split == "overlap_by_one":
        reconstruction_skip = self.model.num_inference_steps - 1
      elif self.model.training_data_split == "no_overlap":
        reconstruction_skip = self.model.num_inference_steps
      elif self.model.training_data_split == "include_inference":
        reconstruction_skip = 0
      else:
        raise NotImplementedError()
    else:
      reconstruction_skip = 1

    full_forward_targets = jax.tree_map(
        lambda x: x[:, :, reconstruction_skip:], full_trajectory)
    full_backward_targets = jax.tree_map(
        lambda x: x[:, :, :x.shape[2] - reconstruction_skip], full_trajectory)
    train_targets_length = (self.model.train_sequence_length -
                            reconstruction_skip)
    full_targets_length = full_forward_targets.shape[2]
    # Fully unroll the model and reconstruct the whole sequence
    full_prediction, full_latents = self._get_reconstructions(self._params, full_trajectory,
                                                rng_key, prefix == "forward",
                                                True)
    # Take the mean of the prediction distribution for the reconstruction
    full_prediction = full_prediction.mean()

    full_targets = (full_forward_targets if prefix == "forward" else
                    full_backward_targets)

    # In cases where the model can run backwards it is possible to reconstruct
    # parts which were indented to be skipped, so here we take care of that.
    if full_prediction.shape[2] > full_targets_length:
      if prefix == "forward":
        full_prediction, full_latents = jax.tree_map(
            lambda x: x[:, :, -full_targets_length:], (full_prediction, full_latents))
      else:
        full_prediction, full_latents = jax.tree_map(
            lambda x: x[:, :, :full_targets_length], (full_prediction, full_latents))

    # Based on the prefix and suffix fetch correct predictions and targets
    if prefix == "forward" and suffix == "train":
      predict, targets, latents = jax.tree_map(
          lambda x: x[:, :, :train_targets_length],
          (full_prediction, full_targets, full_latents))
    elif prefix == "forward" and suffix == "extrapolation":
      predict, targets, latents = jax.tree_map(
          lambda x: x[:, :, train_targets_length:],
          (full_prediction, full_targets, full_latents))
    elif prefix == "backward" and suffix == "train":
      predict, targets, latents = jax.tree_map(
          lambda x: x[:, :, -train_targets_length:],
          (full_prediction, full_targets, full_latents))
    elif prefix == "backward" and suffix == "extrapolation":
      predict, targets, latents = jax.tree_map(
          lambda x: x[:, :, :-train_targets_length],
          (full_prediction, full_targets, full_latents))
    else:
      predict, targets, latents = full_prediction, full_targets, full_latents

    return predict, targets, latents

  def _initialize_eval(self):
    length = (self.model.train_sequence_length +
              self.config.num_extrapolation_steps)
    batch_size = self.config.evaluation.batch_size
    self._eval_input = load_datasets.dataset_as_iter(
        load_datasets.load_dataset,
        path=self.config.dataset_folder,
        tfrecord_prefix="test",
        sub_sample_length=length,
        per_device_batch_size=batch_size,
        num_epochs=1,
        drop_remainder=False,
        multi_device=True,
        shuffle=False,
        cache=False,
        keys_to_preserve=["image"]
    )
    self._eval_batch = jax.pmap(
        self._jax_eval_step_fn, axis_name="i")
    self._get_reconstructions = jax.pmap(
        self.model.reconstruct, axis_name="i",
        static_broadcasted_argnums=(3, 4))
    if isinstance(self.model,
                  common.deterministic_vae.DeterministicLatentsGenerativeModel):
      self._get_samples = jax.pmap(
          self.model.sample_trajectories_from_prior,
          static_broadcasted_argnums=(1, 3, 4))

  def _initialize_eval_metric(self):
    self._eval_input_metric = utils.py_prefetch(
        load_datasets.dataset_as_iter(
            load_datasets.load_dataset,
            path=self.config.dataset_folder,
            tfrecord_prefix="test",
            sub_sample_length=None,
            per_device_batch_size=self.config.evaluation_metric.batch_size,
            multi_device=True,
            num_epochs=None,
            drop_remainder=False,
            cache=False,
            shuffle=False,
            keys_to_preserve=["image", "x"]
        )
    )
    def compute_gt_state_and_latents(*args):
      # Note that the `dt` has to be passed as a kwargs argument
      if len(args) == 4:
        return self.model.gt_state_and_latents(*args[:4])
      elif len(args) == 5:
        return self.model.gt_state_and_latents(*args[:4], dt=args[4])
      else:
        raise NotImplementedError()
    self._compute_gt_state_and_latents = jax.pmap(
        compute_gt_state_and_latents, static_broadcasted_argnums=3)

  def _initialize_eval_vpt(self):
    dataset_name = self.config.dataset_folder.split("/")[-1]
    dataset_folder = self.config.dataset_folder
    if dataset_name in ("hnn_mass_spring_dt_0_05",
                        "mass_spring_colors_v1_dt_0_05",
                        "hnn_pendulum_dt_0_05",
                        "pendulum_colors_v1_dt_0_05",
                        "matrix_rps_dt_0_1",
                        "matrix_mp_dt_0_1"):
      dataset_folder += "_long_trajectory"

    self._eval_input_vpt = utils.py_prefetch(
        load_datasets.dataset_as_iter(
            load_datasets.load_dataset,
            path=dataset_folder,
            tfrecord_prefix="test",
            sub_sample_length=None,
            per_device_batch_size=self.config.evaluation_vpt.batch_size,
            multi_device=True,
            num_epochs=None,
            drop_remainder=False,
            cache=False,
            shuffle=False,
            keys_to_preserve=["image", "x"]
        )
    )

    self._get_reconstructions = jax.pmap(
        self.model.reconstruct, axis_name="i",
        static_broadcasted_argnums=(3, 4))

  def _jax_eval_step_fn(self, params, state, rng_key, batch, step):
    # We care only about the statistics
    _, (_, stats, _) = self.model.training_objectives(params, state, rng_key,
                                                      batch, step,
                                                      is_training=False)
    # Compute the full batch size
    batch_size = jax.tree_flatten(batch)[0][0].shape[0]
    batch_size = utils.psum_if_pmap(batch_size, axis_name="i")

    return self._process_stats(stats, axis_name="i"), batch_size

  def _eval_batch_vpt(self, params, state, rng_key, batch, log_gt=True, plot_latents=True):
    full_trajectory = utils.extract_image(batch)
    prefixes = ("forward",
                "backward") if self.model.can_run_backwards else ("forward",)
    stats = dict()
    movies = dict()
    vpt_abs_scores = []
    vpt_rel_scores = []
    seq_length = None
    for prefix in prefixes:
      reconstruction, gt_images, latents = self._reconstruct_and_align(
          rng_key, full_trajectory, prefix, "extrapolation")
      seq_length = gt_images.shape[2]

      # reconstruction_pr, gt_images_pr, latents_pr = self._reconstruct_and_align(
      #     rng_key, full_trajectory, prefix, "extrapolation")

      mse_norm = np.mean(
          (gt_images - reconstruction)**2, axis=(3, 4, 5)) / np.mean(
              gt_images**2, axis=(3, 4, 5))

      vpt_scores = []
      for i in range(mse_norm.shape[1]):
        vpt_ind = np.argwhere(mse_norm[:, i:i + 1, :] > self.config.evaluation_vpt.vpt_threshold)

        if vpt_ind.shape[0] > 0:
          vpt_ind = vpt_ind[0][2] # If any machine failed, take their first index... 
                                  # maybe should chcek for consistent machine_id_IDX 
                                  # (Only matters in perfect prediction case)
        else:
          vpt_ind = mse_norm.shape[-1] # Otherwise report perfect

        vpt_scores.append(vpt_ind)

      vpt_abs_scores.append(np.median(vpt_scores))
      vpt_rel_scores.append(np.median(vpt_scores) / seq_length)
      scores = {"vpt_abs": vpt_abs_scores[-1], "vpt_rel": vpt_rel_scores[-1]}
      scores = utils.to_numpy(scores)
      scores = utils.filter_only_scalar_stats(scores)
      stats[prefix] = scores

      if 'spatial_dim' in self.model.latent_system_kwargs:
        if self.model.latent_dynamics_type == 'cornn':
          # If we only decode the z-position component, that is all that is passed
          n_components = 2 # Currently both pos & velocity (else change to 1)
        else:
          n_components = self.model.latent_system_kwargs['n_components']
        latent_shape = (self.model.latent_system_kwargs['spatial_dim'], 
                        self.model.latent_system_kwargs['spatial_dim'],
                        self.model.latent_system_kwargs['n_state_channels'],
                        n_components)
        n_state_channels = self.model.latent_system_kwargs['n_state_channels']
      elif 'spatial_height' in self.model.latent_system_kwargs:
        if self.model.latent_dynamics_type == 'cornn':
          # If we only decode the z-position component, that is all that is passed
          n_components = 2 # Currently both pos & velocity (else change to 1)
        else:
          n_components = self.model.latent_system_kwargs['n_components']
        latent_shape = (self.model.latent_system_kwargs['spatial_height'], 
                        self.model.latent_system_kwargs['spatial_width'],
                        self.model.latent_system_kwargs['n_state_channels'],
                        n_components)
        n_state_channels = self.model.latent_system_kwargs['n_state_channels']
      else:
        latent_shape = (1, self.model.latent_system_dim, 1, 1)
        n_components = 1
        n_state_channels = 1
      
      # recon tensors are shape (T, H, W, C), need to swap to (T, C, H, W) and normalize to int(0,255) and flip vertical axis
      movies[prefix] = {"reconstructions": utils.normalize_int(utils.to_numpy(reconstruction[0, 0]).swapaxes(1, 3).swapaxes(2,3)[:,:,::-1])}
      # movies[prefix] = {"reconstructions_phaseroll": utils.normalize_int(utils.to_numpy(reconstruction_pr[0, 0]).swapaxes(1, 3).swapaxes(2,3)[:,:,::-1])}

      if log_gt:
        movies[prefix]['gt_images'] = utils.normalize_int(utils.to_numpy(gt_images[0, 0].swapaxes(1, 3).swapaxes(2,3)[:,:,::-1]))
      if plot_latents:
        if self.model.latent_system_net_type == "mlp":
          # latent shape (Devices, B, T, XV*H*W)
          assert latents.shape[3] == np.prod(latent_shape)
        if self.model.latent_system_net_type == "conv":
          # latent tensors are shape (Devices, B, T, H, W, XV)
          assert latents.shape[3] == latents.shape[4]
          assert latents.shape[5] == self.model.latent_system_kwargs['n_components']

        spatial_latents = utils.to_numpy(latents[0, 0, :, :]).reshape(-1, 1, *latent_shape)
        # spatial_latents_pr = utils.to_numpy(latents_pr[0, 0, :, :]).reshape(-1, 1, *latent_shape)


        for sc in range(n_state_channels):
          if n_components == 1:
            position_spatial_latents = spatial_latents[:, :, :, :, sc, 0]
            movies[prefix][f"position_latents_{sc}"] =  utils.normalize_int(position_spatial_latents)
            movies[prefix][f"position_latents_phase_{sc}"] = utils.normalize_int(utils.to_numpy(utils.rgb_phase(position_spatial_latents)))

          elif n_components == 2:
            position_spatial_latents = spatial_latents[:, :, :, :, sc, 0]
            momentum_spatial_latents = spatial_latents[:, :, :, :, sc, 1]
            movies[prefix][f"position_latents_{sc}"] =  utils.normalize_int(position_spatial_latents)
            movies[prefix][f"position_latents_phase_{sc}"] = utils.normalize_int(utils.to_numpy(utils.rgb_phase(position_spatial_latents)))

            movies[prefix][f"momentum_latents_{sc}"] =  utils.normalize_int(momentum_spatial_latents)

            # position_spatial_latents_pr = spatial_latents_pr[:, :, :, :, sc, 0]
            # momentum_spatial_latents_pr = spatial_latents_pr[:, :, :, :, sc, 1]
            # movies[prefix][f"position_latents_PhaseRoll_{sc}"] =  utils.normalize_int(position_spatial_latents_pr)
            # movies[prefix][f"position_latents_PhaseRoll_phase_{sc}"] = utils.normalize_int(utils.to_numpy(utils.rgb_phase(position_spatial_latents_pr)))

            # movies[prefix][f"momentum_latents_PhaseRoll_{sc}"] =  utils.normalize_int(momentum_spatial_latents_pr)

          elif n_components == 3:
            vx_spatial_latents = spatial_latents[:, :, :, :, sc, 0]
            vy_spatial_latents = spatial_latents[:, :, :, :, sc, 1]
            rho_spatial_latents = spatial_latents[:, :, :, :, sc, 2]
            movies[prefix][f"vx_latents_{sc}"] =  utils.normalize_int(vx_spatial_latents)
            movies[prefix][f"vy_latents_{sc}"] =  utils.normalize_int(vy_spatial_latents)
            movies[prefix][f"rho_latents_{sc}"] =  utils.normalize_int(rho_spatial_latents)
          else:
            raise NotImplementedError

    if self.model.latent_dynamics_type == 'CO' and 'CoupledOscillator' in params:
      stats['alpha'] = utils.get_first(params['CoupledOscillator']['alpha']) if 'alpha' in params['CoupledOscillator'] else 0.0
      stats['gamma'] = utils.get_first(params['CoupledOscillator']['gamma']) if 'gamma' in params['CoupledOscillator'] else 0.0
      stats['W_bar'] = utils.get_first(params['CoupledOscillator']['W_bar']) if 'W_bar' in params['CoupledOscillator'] else 0.0
      stats['b_bar'] = utils.get_first(params['CoupledOscillator']['b_bar']) if 'b_bar' in params['CoupledOscillator'] else 0.0

    elif self.model.latent_dynamics_type == 'cornn':
      pname = 'CORNN_0'
      if pname in params:
        stats['alpha'] = utils.get_first(params[pname]['alpha']) if 'alpha' in params[pname] else 0.0
        stats['gamma'] = utils.get_first(params[pname]['gamma']) if 'gamma' in params[pname] else 0.0
        stats['b_bar'] = utils.get_first(params[pname]['b_bar']) if 'b_bar' in params[pname] else 0.0
        stats['dt'] = utils.get_first(params[pname]['dt']) if 'dt' in params[pname] else 0.0

    # import ipdb; ipdb.set_trace()
    stats['pct_waves'] = utils.percent_waves(position_spatial_latents, rng_key)

    stats["vpt_abs"] = utils.to_numpy(np.mean(vpt_abs_scores))
    stats["vpt_rel"] = utils.to_numpy(np.mean(vpt_rel_scores))
    logging.info("vpt_abs: %s, seq_length: %d}",
                 str(vpt_abs_scores), seq_length)
    return stats, movies

  def _eval_batch_metric(self, params, rng, batch, eval_seq_len=200):
    # Initialise alpha values for Lasso regression
    alpha_sweep = np.logspace(self.config.evaluation_metric.alpha_min_logspace,
                              self.config.evaluation_metric.alpha_max_logspace,
                              self.config.evaluation_metric.alpha_step_n)
    trajectory_n = self.config.evaluation_metric.batch_size
    subsection = f"{trajectory_n}tr"
    stats = dict()

    # Get data
    (gt_trajectory,
     model_trajectory,
     informative_dim_n) = self._get_gt_and_model_phase_space_for_eval(
         params, rng, batch, eval_seq_len)

    # import ipdb; ipdb.set_trace()

    # Calculate SyMetric scores
    if informative_dim_n > 1:
      scores, *_ = eval_metric.calculate_symetric_score(
          gt_trajectory,
          model_trajectory,
          self.config.evaluation_metric.max_poly_order,
          self.config.evaluation_metric.max_jacobian_score,
          self.config.evaluation_metric.rsq_threshold,
          self.config.evaluation_metric.sym_threshold,
          self.config.evaluation_metric.evaluation_point_n,
          trajectory_n=trajectory_n,
          weight_tolerance=self.config.evaluation_metric.weight_tolerance,
          alpha_sweep=alpha_sweep,
          max_iter=self.config.evaluation_metric.max_iter,
          cv=self.config.evaluation_metric.cv)

      scores["unmasked_latents"] = informative_dim_n
      scores = utils.to_numpy(scores)
      scores = utils.filter_only_scalar_stats(scores)
      stats[subsection] = scores
    else:
      scores = {
          "poly_exp_order":
              self.config.evaluation_metric.max_poly_order,
          "rsq":
              0,
          "sym":
              self.config.evaluation_metric.max_jacobian_score,
          "SyMetric": 0.0,
          "unmasked_latents":
              informative_dim_n
      }
      scores = utils.to_numpy(scores)
      scores = utils.filter_only_scalar_stats(scores)
      stats[subsection] = scores

    return stats

  def _get_gt_and_model_phase_space_for_eval(self, params, rng, batch,
                                             eval_seq_len):
    # Get data
    gt_data, model_data, z0 = utils.stack_device_dim_into_batch(
        self._compute_gt_state_and_latents(params, rng, batch, eval_seq_len)
    )

    if isinstance(self.model, AutoregressiveModel):
      # These models return the `z` for the whole sequence
      z0 = z0[:, 0]

    # If latent space is image like, reshape it down to vector
    if self.model.latent_system_net_type == "conv":
      z0 = jax.tree_map(utils.reshape_latents_conv_to_flat, z0)
      model_data = jax.tree_map(
          lambda x: utils.reshape_latents_conv_to_flat(x, axis_n_to_keep=2),
          model_data)

    # Create mask to get rid of uninformative latents
    latent_mask = eval_metric.create_latent_mask(z0)
    informative_dim_n = np.sum(latent_mask)

    if self.model.latent_dynamics_type == 'cornn':
      model_data = model_data[:, :, :model_data.shape[2]//2]
    model_data = model_data[:, :, latent_mask]
    logging.info("Masking out model data, leaving dim_n=%d dimensions.",
                 model_data.shape[-1])

    gt_trajectory = np.reshape(
        gt_data,
        [np.product(gt_data.shape[:-1]), gt_data.shape[-1]]
        )

    model_trajectory = np.reshape(model_data, [
        np.product(model_data.shape[:-1]), model_data.shape[-1]
    ])

    # Standardize data
    gt_trajectory = eval_metric.standardize_data(gt_trajectory)
    model_trajectory = eval_metric.standardize_data(model_trajectory)

    return gt_trajectory, model_trajectory, informative_dim_n
