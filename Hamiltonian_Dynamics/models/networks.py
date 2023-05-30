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
"""Module containing all of the networks as Haiku modules."""
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from absl import logging
import distrax
import haiku as hk
import jax
import jax.numpy as jnp

from physics_inspired_models import utils

Activation = Union[str, Callable[[jnp.ndarray], jnp.ndarray]]


class DenseNet(hk.Module):
  """A feed forward network (MLP)."""

  def __init__(
      self,
      num_units: Sequence[int],
      activate_final: bool = False,
      activation: Activation = "leaky_relu",
      name: Optional[str] = None):
    super().__init__(name=name)
    self.num_units = num_units
    self.num_layers = len(self.num_units)
    self.activate_final = activate_final
    self.activation = utils.get_activation(activation)

    self.linear_modules = []
    for i in range(self.num_layers):
      self.linear_modules.append(
          hk.Linear(
              output_size=self.num_units[i],
              name=f"ff_{i}"
          )
      )

  def __call__(self, inputs: jnp.ndarray, is_training: bool):
    net = inputs
    for i, linear in enumerate(self.linear_modules):
      net = linear(net)
      if i < self.num_layers - 1 or self.activate_final:
        net = self.activation(net)
    return net


class Conv2DNet(hk.Module):
  """Convolutional Network."""

  def __init__(
      self,
      output_channels: Sequence[int],
      kernel_shapes: Union[int, Sequence[int]] = 3,
      strides: Union[int, Sequence[int]] = 1,
      padding: Union[str, Sequence[str]] = "SAME",
      data_format: str = "NHWC",
      with_batch_norm: bool = False,
      activate_final: bool = False,
      activation: Activation = "leaky_relu",
      name: Optional[str] = None):
    super().__init__(name=name)
    self.output_channels = tuple(output_channels)
    self.num_layers = len(self.output_channels)
    self.kernel_shapes = utils.bcast_if(kernel_shapes, int, self.num_layers)
    self.strides = utils.bcast_if(strides, int, self.num_layers)
    self.padding = utils.bcast_if(padding, str, self.num_layers)
    self.data_format = data_format
    self.with_batch_norm = with_batch_norm
    self.activate_final = activate_final
    self.activation = utils.get_activation(activation)

    if len(self.kernel_shapes) != self.num_layers:
      raise ValueError(f"Kernel shapes is of size {len(self.kernel_shapes)}, "
                       f"while output_channels is of size{self.num_layers}.")
    if len(self.strides) != self.num_layers:
      raise ValueError(f"Strides is of size {len(self.kernel_shapes)}, while "
                       f"output_channels is of size{self.num_layers}.")
    if len(self.padding) != self.num_layers:
      raise ValueError(f"Padding is of size {len(self.padding)}, while "
                       f"output_channels is of size{self.num_layers}.")

    self.conv_modules = []
    self.bn_modules = []
    for i in range(self.num_layers):
      self.conv_modules.append(
          hk.Conv2D(
              output_channels=self.output_channels[i],
              kernel_shape=self.kernel_shapes[i],
              stride=self.strides[i],
              padding=self.padding[i],
              data_format=data_format,
              name=f"conv_2d_{i}")
      )
      if with_batch_norm:
        self.bn_modules.append(
            hk.BatchNorm(
                create_offset=True,
                create_scale=False,
                decay_rate=0.999,
                name=f"batch_norm_{i}")
        )
      else:
        self.bn_modules.append(None)

  def __call__(self, inputs: jnp.ndarray, is_training: bool):
    assert inputs.ndim == 4
    net = inputs
    for i, (conv, bn) in enumerate(zip(self.conv_modules, self.bn_modules)):
      net = conv(net)
      # Batch norm
      if bn is not None:
        net = bn(net, is_training=is_training)
      if i < self.num_layers - 1 or self.activate_final:
        net = self.activation(net)
    return net


class SpatialConvEncoder(hk.Module):
  """Spatial Convolutional Encoder for learning the Hamiltonian."""

  def __init__(
      self,
      latent_dim: int,
      conv_channels: Union[Sequence[int], int],
      num_blocks: int,
      blocks_depth: int = 2,
      distribution_name: str = "diagonal_normal",
      aggregation_type: Optional[str] = None,
      data_format: str = "NHWC",
      activation: Activation = "leaky_relu",
      scale_factor: int = 2,
      kernel_shapes: Union[Sequence[int], int] = 3,
      padding: Union[Sequence[str], str] = "SAME",
      name: Optional[str] = None):
    super().__init__(name=name)
    if aggregation_type not in (None, "max", "mean", "linear_projection"):
      raise ValueError(f"Unrecognized aggregation_type={aggregation_type}.")
    self.latent_dim = latent_dim
    self.conv_channels = conv_channels
    self.num_blocks = num_blocks
    self.scale_factor = scale_factor
    self.data_format = data_format
    self.distribution_name = distribution_name
    self.aggregation_type = aggregation_type

    # Compute the required size of the output
    if distribution_name is None:
      self.output_dim = latent_dim
    elif distribution_name == "diagonal_normal":
      self.output_dim = 2 * latent_dim
    else:
      raise ValueError(f"Unrecognized distribution_name={distribution_name}.")

    if isinstance(conv_channels, int):
      conv_channels = [[conv_channels] * blocks_depth
                       for _ in range(num_blocks)]
      conv_channels[-1] += [self.output_dim]
    else:
      assert isinstance(conv_channels, (list, tuple))
      assert len(conv_channels) == num_blocks
      conv_channels = list(list(c) for c in conv_channels)
      conv_channels[-1].append(self.output_dim)

    if isinstance(kernel_shapes, tuple):
      kernel_shapes = list(kernel_shapes)

    # Convolutional blocks
    self.blocks = []
    for i, channels in enumerate(conv_channels):
      if isinstance(kernel_shapes, int):
        extra_kernel_shapes = 0
      else:
        extra_kernel_shapes = [3] * (len(channels) - len(kernel_shapes))

      self.blocks.append(Conv2DNet(
          output_channels=channels,
          kernel_shapes=kernel_shapes + extra_kernel_shapes,
          strides=[self.scale_factor] + [1] * (len(channels) - 1),
          padding=padding,
          data_format=data_format,
          with_batch_norm=False,
          activate_final=i < num_blocks - 1,
          activation=activation,
          name=f"block_{i}"
      ))

  def spatial_aggregation(self, x: jnp.ndarray) -> jnp.ndarray:
    if self.aggregation_type is None:
      return x
    axis = (1, 2) if self.data_format == "NHWC" else (2, 3)
    if self.aggregation_type == "max":
      return jnp.max(x, axis=axis)
    if self.aggregation_type == "mean":
      return jnp.mean(x, axis=axis)
    if self.aggregation_type == "linear_projection":
      x = x.reshape(x.shape[:-3] + (-1,))
      return hk.Linear(self.output_dim, name="LinearProjection")(x)
    raise NotImplementedError()

  def make_distribution(self, net_output: jnp.ndarray) -> distrax.Distribution:
    if self.distribution_name is None:
      return net_output
    elif self.distribution_name == "diagonal_normal":
      if self.aggregation_type is None:
        split_axis, num_axes = self.data_format.index("C"), 3
      else:
        split_axis, num_axes = 1, 1
      # Add an extra axis if the input has more than 1 batch dimension
      split_axis += net_output.ndim - num_axes - 1
      loc, log_scale = jnp.split(net_output, 2, axis=split_axis)
      return distrax.Normal(loc, jnp.exp(log_scale))
    else:
      raise NotImplementedError()

  def __call__(
      self,
      inputs: jnp.ndarray,
      is_training: bool
  ) -> Union[jnp.ndarray, distrax.Distribution]:
    # Treat any extra dimensions (like time) as the batch
    batched_shape = inputs.shape[:-3]
    net = jnp.reshape(inputs, (-1,) + inputs.shape[-3:])

    # Apply all blocks in sequence
    for block in self.blocks:
      net = block(net, is_training=is_training)

    # Final projection
    net = self.spatial_aggregation(net)

    # Reshape back to correct dimensions (like batch + time)
    net = jnp.reshape(net, batched_shape + net.shape[1:])

    # Return a distribution over the observations
    return self.make_distribution(net)


class SpatialConvDecoder(hk.Module):
  """Spatial Convolutional Decoder for learning the Hamiltonian."""

  def __init__(
      self,
      initial_spatial_shape: Sequence[int],
      conv_channels: Union[Sequence[int], int],
      num_blocks: int,
      max_de_aggregation_dims: int,
      blocks_depth: int = 2,
      scale_factor: int = 2,
      output_channels: int = 3,
      h_const_channels: int = 2,
      data_format: str = "NHWC",
      activation: Activation = "leaky_relu",
      learned_sigma: bool = False,
      de_aggregation_type: Optional[str] = None,
      final_activation: Activation = "sigmoid",
      discard_half_de_aggregated: bool = False,
      kernel_shapes: Union[Sequence[int], int] = 3,
      padding: Union[Sequence[str], str] = "SAME",
      name: Optional[str] = None):
    super().__init__(name=name)
    if de_aggregation_type not in (None, "tile", "linear_projection"):
      raise ValueError(f"Unrecognized de_aggregation_type="
                       f"{de_aggregation_type}.")
    self.num_blocks = num_blocks
    self.scale_factor = scale_factor
    self.h_const_channels = h_const_channels
    self.data_format = data_format
    self.learned_sigma = learned_sigma
    self.initial_spatial_shape = tuple(initial_spatial_shape)
    self.final_activation = utils.get_activation(final_activation)
    self.de_aggregation_type = de_aggregation_type
    self.max_de_aggregation_dims = max_de_aggregation_dims
    self.discard_half_de_aggregated = discard_half_de_aggregated

    if isinstance(conv_channels, int):
      conv_channels = [[conv_channels] * blocks_depth
                       for _ in range(num_blocks)]
      conv_channels[-1] += [output_channels]
    else:
      assert isinstance(conv_channels, (list, tuple))
      assert len(conv_channels) == num_blocks
      conv_channels = list(list(c) for c in conv_channels)
      conv_channels[-1].append(output_channels)

    # Convolutional blocks
    self.blocks = []
    for i, channels in enumerate(conv_channels):
      is_final_block = i == num_blocks - 1
      self.blocks.append(
          Conv2DNet(  # pylint: disable=g-complex-comprehension
              output_channels=channels,
              kernel_shapes=kernel_shapes,
              strides=1,
              padding=padding,
              data_format=data_format,
              with_batch_norm=False,
              activate_final=not is_final_block,
              activation=activation,
              name=f"block_{i}"
          ))

  def spatial_de_aggregation(self, x: jnp.ndarray) -> jnp.ndarray:
    if self.de_aggregation_type is None:
      assert x.ndim >= 4
      if self.data_format == "NHWC":
        assert x.shape[1:3] == self.initial_spatial_shape
      elif self.data_format == "NCHW":
        assert x.shape[2:4] == self.initial_spatial_shape
      return x
    elif self.de_aggregation_type == "linear_projection":
      assert x.ndim == 2
      n, d = x.shape
      d = min(d, self.max_de_aggregation_dims or d)
      out_d = d * self.initial_spatial_shape[0] * self.initial_spatial_shape[1]
      x = hk.Linear(out_d, name="LinearProjection")(x)
      if self.data_format == "NHWC":
        shape = (n,) + self.initial_spatial_shape + (d,)
      else:
        shape = (n, d) + self.initial_spatial_shape
      return x.reshape(shape)
    elif self.de_aggregation_type == "tile":
      assert x.ndim == 2
      if self.data_format == "NHWC":
        repeats = (1,) + self.initial_spatial_shape + (1,)
        x = x[:, None, None, :]
      else:
        repeats = (1, 1) + self.initial_spatial_shape
        x = x[:, :, None, None]
      return jnp.tile(x, repeats)
    else:
      raise NotImplementedError()

  def add_constant_channels(self, inputs: jnp.ndarray) -> jnp.ndarray:
    # --------------------------------------------
    # This is purely for TF compatibility purposes
    if self.discard_half_de_aggregated:
      axis = self.data_format.index("C")
      inputs, _ = jnp.split(inputs, 2, axis=axis)
    # --------------------------------------------

    # An extra constant channels
    if self.data_format == "NHWC":
      h_shape = self.initial_spatial_shape + (self.h_const_channels,)
    else:
      h_shape = (self.h_const_channels,) + self.initial_spatial_shape
    h_const = hk.get_parameter("h", h_shape, dtype=inputs.dtype,
                               init=hk.initializers.Constant(1))
    h_const = jnp.tile(h_const, reps=[inputs.shape[0], 1, 1, 1])
    return jnp.concatenate([h_const, inputs], axis=self.data_format.index("C"))

  def make_distribution(self, net_output: jnp.ndarray) -> distrax.Distribution:
    if self.learned_sigma:
      init = hk.initializers.Constant(- jnp.log(2.0) / 2.0)
      log_scale = hk.get_parameter("log_scale", shape=(),
                                   dtype=net_output.dtype, init=init)
      scale = jnp.full_like(net_output, jnp.exp(log_scale))
    else:
      scale = jnp.full_like(net_output, 1 / jnp.sqrt(2.0))

    return distrax.Normal(net_output, scale)

  def __call__(
      self,
      inputs: jnp.ndarray,
      is_training: bool
  ) -> distrax.Distribution:
    # Apply the spatial de-aggregation
    inputs = self.spatial_de_aggregation(inputs)

    # Add the parameterized constant channels
    net = self.add_constant_channels(inputs)

    # Apply all the blocks
    for block in self.blocks:
      # Up-sample the image
      net = utils.nearest_neighbour_upsampling(net, self.scale_factor)
      # Apply the convolutional block
      net = block(net, is_training=is_training)

    # Apply any specific output nonlinearity
    net = self.final_activation(net)

    # Construct the distribution over the observations
    return self.make_distribution(net)





class LocallyConnectedLinear(hk.Module):
  """Locally Connected Linear Layer
  
  Each location has a set of weights and biases which   
  """

  def __init__(
      self,
      input_channels: int,
      output_channels: int,
      output_size: Union[int, Sequence[int]],
      kernel_size: Union[int, Sequence[int]] = 3,
      stride: Union[int, Sequence[int]] = 1,
      padding: Union[str, Sequence[str]] = "VALID",

      w_init: Any = hk.initializers.TruncatedNormal(), # lambda s, dtype: jnp.full(s, 1.0/9.0, dtype),
      data_format: str = "NHWC",
      name: Optional[str] = None):
    super().__init__(name=name)
    self.input_channels = input_channels
    self.output_channels = output_channels
    self.output_size = utils.bcast_if(output_size, int, 2)
    self.kernel_size = utils.bcast_if(kernel_size, int, 2)
    self.stride = utils.bcast_if(stride, int, 2)
    self.padding = padding
    self.data_format = data_format
    self.w_init = w_init
    self.lname = name

  def __call__(self, inputs: jnp.ndarray, is_training: bool=True):
    lc_params = hk.get_parameter(self.lname + "_lc_params", shape=[self.output_size[0], 
                                                         self.output_size[1], 
                                                         self.input_channels * self.kernel_size[0] * self.kernel_size[1],
                                                         self.output_channels],
                                    init=self.w_init)
    x_unf = jax.lax.conv_general_dilated_patches(lhs=inputs, filter_shape = self.kernel_size, 
                        window_strides = self.stride, padding = self.padding, 
                        dimension_numbers  = ('NHWC', 'HWIO', 'NHWC'))   # Out = (N, H, W, KH*KW*IC) 
    x_unf_e = jnp.expand_dims(x_unf, -1)## N, H, W, KH*KW*IC, OutC
    out = (x_unf_e * lc_params).sum([3]) # Sum over in-channels and kernel HW dims --> (N, H, W, OC)

    return out



def add_batch(nest, batch_size: Optional[int]):
  """Adds a batch dimension at axis 0 to the leaves of a nested structure."""
  broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
  return jax.tree_map(broadcast, nest)

# Original CoRNN

class CoRNN(hk.RNNCore):
  """Couple Oscillator RNN with spatial latent state and locally connected layers
  """
  def __init__(
    self,
    spatial_height: int,
    spatial_width: int,
    n_components: int,
    n_state_channels: int,
    n_input_channels: int,
    kernel_shape: int,
    sigma: Activation,
    gamma_init: float,
    alpha_init: float,
    b_bar_init: float,
    dt_init: float,
    name: Optional[str] = None
  ):
    super().__init__(name=name)
    self.spatial_height = spatial_height
    self.spatial_width = spatial_width
    self.n_components = n_components
    self.n_state_channels = n_state_channels
    self.n_input_channels = n_input_channels
    pad = kernel_shape // 2
    self.pad_shape = ((0, 0), (pad, pad), (pad, pad), (0, 0))
    self.gamma_init = lambda s, dt: jnp.full(s, gamma_init, dt)
    self.alpha_init = lambda s, dt: jnp.full(s, alpha_init, dt)
    self.dt_init = lambda s, dt: jnp.full(s, dt_init, dt)
    self.b_bar_init = lambda s, dt: jnp.full(s, b_bar_init, dt)
    if sigma == 'identity':
      self.sigma = lambda x: x
    else:
      self.sigma = utils.get_activation(sigma)

    constant_k = 1.0/9.0 # jnp.array([[0,0,0],[0,1,0],[0,0,0]]).reshape((3,3,1,1))
    ones_state = jnp.ones((kernel_shape, kernel_shape, self.n_state_channels, self.n_state_channels)) * 1.0/9.0
    # ones_state_indep = jnp.ones((kernel_shape, kernel_shape, 1, self.n_state_channels)) * 1.0/9.0
    ones_input = jnp.ones((kernel_shape, kernel_shape, self.n_input_channels, self.n_state_channels)) * 1.0/9.0

    # ones_state = jnp.ones((self.spatial_dim, self.spatial_dim, self.n_state_channels * kernel_shape ** 2, self.n_state_channels)) * 1.0/9.0
    # ones_input = jnp.ones((self.spatial_dim, self.spatial_dim, self.n_input_channels * kernel_shape ** 2, self.n_state_channels)) * 1.0/9.0
    coupling_init = lambda s, dt: jnp.full(s, ones_state, dt) + hk.initializers.TruncatedNormal(stddev=0.1)(s, dt)
    wbar_init = lambda s, dt: jnp.full(s, ones_state, dt) + hk.initializers.TruncatedNormal(stddev=0.1)(s, dt)
    v_init = lambda s, dt: jnp.full(s, ones_input, dt) + hk.initializers.TruncatedNormal(stddev=0.1)(s, dt)

    self.coupling = hk.Conv2D(output_channels=self.n_state_channels,
                              kernel_shape=kernel_shape,
                              stride=1,
                              padding='VALID',
                              # w_init=hk.initializers.Constant(constant_k),
                              w_init=coupling_init,
                              # feature_group_count=self.n_state_channels, ## Independant Channels
                              with_bias=False,
                              data_format='NHWC',
                              name="CouplingNet")

    # self.coupling = LocallyConnectedLinear(input_channels=self.n_state_channels,
    #                                         output_channels=self.n_state_channels,
    #                                         output_size=self.spatial_dim,
    #                                         kernel_size=kernel_shape,
    #                                         stride=1,
    #                                         padding="VALID",
    #                                         w_init=coupling_init,
    #                                         name="Coupling_LC_Net")

    self.W_bar = hk.Conv2D(output_channels=self.n_state_channels,
                            kernel_shape=kernel_shape,
                            stride=1,
                            padding='VALID',
                            # w_init=hk.initializers.Constant(constant_k),
                            w_init=wbar_init,
                            # feature_group_count=self.n_state_channels, ## Independant Channels
                            with_bias=False,
                            data_format='NHWC',
                            name="DampingCouplingNet")

    # self.W_bar = LocallyConnectedLinear(input_channels=self.n_state_channels,
    #                                     output_channels=self.n_state_channels,
    #                                     output_size=self.spatial_dim,
    #                                     kernel_size=kernel_shape,
    #                                     stride=1,
    #                                     padding="VALID",
    #                                     w_init=wbar_init,
    #                                     name="Damping_LC_Net")

    # self.W_bar = lambda x: x

    self.V = hk.Conv2D(output_channels=self.n_state_channels,
                        kernel_shape=kernel_shape,
                        stride=1,
                        padding='VALID',
                        # w_init=hk.initializers.Constant(constant_k),
                        w_init=v_init,
                        feature_group_count=1,
                        with_bias=False,
                        data_format='NHWC',
                        name="ForcingCouplingNet")

    # self.V = LocallyConnectedLinear(input_channels=self.n_input_channels,
    #                                 output_channels=self.n_state_channels,
    #                                 output_size=self.spatial_dim,
    #                                 kernel_size=kernel_shape,
    #                                 stride=1,
    #                                 padding="VALID",
    #                                 w_init=v_init,
    #                                 name="Forcing_LC_Net")

    # self.V = lambda x: x

  def __call__(self, inputs, prev_state):
    # coupling_val = hk.get_parameter("coupling", shape=[self.spatial_dim, self.spatial_dim, 1], 
    #                                   init=lambda s, dt: jnp.full(s, 1.0, dt))
    # self.coupling = lambda x: x * coupling_val
    # W_bar_val = hk.get_parameter("W_bar", shape=[self.spatial_dim, self.spatial_dim, 1], 
    #                                   init=lambda s, dt: jnp.full(s, 0.0, dt))
    # self.W_bar = lambda x: x * W_bar_val

    gamma_pos = jax.nn.relu(hk.get_parameter("gamma", shape=[], init=self.gamma_init))
    alpha_pos = jax.nn.relu(hk.get_parameter("alpha", shape=[], init=self.alpha_init))
    b_bar = hk.get_parameter("b_bar", shape=[self.spatial_height, self.spatial_width, self.n_state_channels], init=self.b_bar_init)
    dt = jax.nn.sigmoid(hk.get_parameter("dt", shape=[], init=self.dt_init))

    # spatial_inputs = inputs.reshape(-1, self.spatial_dim, self.spatial_dim, self.n_components)
    spatial_inputs = inputs.reshape(-1, self.spatial_height, self.spatial_width, self.n_input_channels)
    spatial_state = prev_state.reshape(-1, self.spatial_height, self.spatial_width, self.n_state_channels, self.n_components)
    x0 = spatial_state[:, :, :, :, 0]
    v0 = spatial_state[:, :, :, :, 1]
    x_pad = jnp.pad(x0, pad_width=self.pad_shape, mode='wrap')
    v_pad = jnp.pad(v0, pad_width=self.pad_shape, mode='wrap')
    u_pad = jnp.pad(spatial_inputs, pad_width=self.pad_shape, mode='wrap')

    v_n = v0 + dt * (self.sigma(self.coupling(x_pad) 
                                  + self.W_bar(v_pad)
                                  # + self.W_bar(v0)
                                  + self.V(u_pad)
                                  # + self.V(spatial_inputs)
                                  + b_bar) 
                        - gamma_pos * x0 - alpha_pos * v0)
    x_n = x0 + dt * v_n
    out_state = jnp.stack([x_n, v_n], -1)
    out_state = out_state.reshape(-1, self.spatial_height * self.spatial_width * self.n_components * self.n_state_channels)
    return out_state, out_state

  def initial_state(self, batch_size: Optional[int]):
    state = jnp.zeros([self.spatial_height * self.spatial_width * self.n_components * self.n_state_channels])
    if batch_size is not None:
      state = add_batch(state, batch_size)
    return state

# Linear CoRNN
  
# class CoRNN(hk.RNNCore):
#   """Couple Oscillator RNN with spatial latent state and locally connected layers
#   """
#   def __init__(
#     self,
#     spatial_dim: int,
#     n_components: int,
#     n_state_channels: int,
#     n_input_channels: int,
#     kernel_shape: int,
#     sigma: Activation,
#     gamma_init: float,
#     alpha_init: float,
#     b_bar_init: float,
#     dt_init: float,
#     name: Optional[str] = None
#   ):
#     super().__init__(name=name)
#     self.spatial_dim = spatial_dim
#     self.n_components = n_components
#     self.n_state_channels = n_state_channels
#     self.n_input_channels = n_input_channels
#     pad = kernel_shape // 2
#     self.pad_shape = ((0, 0), (pad, pad), (pad, pad), (0, 0))
#     self.gamma_init = lambda s, dt: jnp.full(s, gamma_init, dt)
#     self.alpha_init = lambda s, dt: jnp.full(s, alpha_init, dt)
#     self.dt_init = lambda s, dt: jnp.full(s, dt_init, dt)
#     self.b_bar_init = lambda s, dt: jnp.full(s, b_bar_init, dt)
#     if sigma == 'identity':
#       self.sigma = lambda x: x
#     else:
#       self.sigma = utils.get_activation(sigma)

#     # constant_k = 1.0/9.0 # jnp.array([[0,0,0],[0,1,0],[0,0,0]]).reshape((3,3,1,1))
#     # ones_state = jnp.ones((kernel_shape, kernel_shape, self.n_state_channels, self.n_state_channels)) * 1.0/9.0
#     # # ones_state_indep = jnp.ones((kernel_shape, kernel_shape, 1, self.n_state_channels)) * 1.0/9.0
#     # ones_input = jnp.ones((kernel_shape, kernel_shape, self.n_input_channels, self.n_state_channels)) * 1.0/9.0

#     # ones_state = jnp.ones((self.spatial_dim, self.spatial_dim, self.n_state_channels * kernel_shape ** 2, self.n_state_channels)) * 1.0/9.0
#     # ones_input = jnp.ones((self.spatial_dim, self.spatial_dim, self.n_input_channels * kernel_shape ** 2, self.n_state_channels)) * 1.0/9.0
#     # coupling_init = lambda s, dt: jnp.full(s, ones_state, dt) + hk.initializers.TruncatedNormal(stddev=0.1)(s, dt)
#     # wbar_init = lambda s, dt: jnp.full(s, ones_state, dt) + hk.initializers.TruncatedNormal(stddev=0.1)(s, dt)
#     # v_init = lambda s, dt: jnp.full(s, ones_input, dt) + hk.initializers.TruncatedNormal(stddev=0.1)(s, dt)

#     self.coupling = hk.Linear(output_size=self.spatial_dim * self.spatial_dim * n_state_channels,
#                               with_bias=False,
#                               name="CouplingNet")

#     self.W_bar = hk.Linear(output_size=self.spatial_dim * self.spatial_dim * n_state_channels,
#                             with_bias=False,
#                             name="DampingCouplingNet")

#     self.V = hk.Linear(output_size=self.spatial_dim * self.spatial_dim * n_state_channels,
#                         with_bias=False,
#                         name="ForcingCouplingNet")


#   def __call__(self, inputs, prev_state):
#     # coupling_val = hk.get_parameter("coupling", shape=[self.spatial_dim, self.spatial_dim, 1], 
#     #                                   init=lambda s, dt: jnp.full(s, 1.0, dt))
#     # self.coupling = lambda x: x * coupling_val
#     # W_bar_val = hk.get_parameter("W_bar", shape=[self.spatial_dim, self.spatial_dim, 1], 
#     #                                   init=lambda s, dt: jnp.full(s, 0.0, dt))
#     # self.W_bar = lambda x: x * W_bar_val

#     gamma_pos = jax.nn.relu(hk.get_parameter("gamma", shape=[], init=self.gamma_init))
#     alpha_pos = jax.nn.relu(hk.get_parameter("alpha", shape=[], init=self.alpha_init))
#     b_bar = hk.get_parameter("b_bar", shape=[self.spatial_dim * self.spatial_dim * self.n_state_channels], init=self.b_bar_init)
#     dt = jax.nn.sigmoid(hk.get_parameter("dt", shape=[], init=self.dt_init))

#     spatial_inputs = inputs.reshape(-1, self.spatial_dim * self.spatial_dim * self.n_input_channels)
#     spatial_state = prev_state.reshape(-1, self.spatial_dim * self.spatial_dim * self.n_state_channels, self.n_components)
#     x0 = spatial_state[:, :, 0]
#     v0 = spatial_state[:, :, 1]
#     x_pad = x0 # jnp.pad(x0, pad_width=self.pad_shape, mode='wrap')
#     v_pad = v0 # jnp.pad(v0, pad_width=self.pad_shape, mode='wrap')
#     u_pad = spatial_inputs # jnp.pad(spatial_inputs, pad_width=self.pad_shape, mode='wrap')

#     v_n = v0 + dt * (self.sigma(self.coupling(x_pad) 
#                                   + self.W_bar(v_pad)
#                                   # + self.W_bar(v0)
#                                   + self.V(u_pad)
#                                   # + self.V(spatial_inputs)
#                                   + b_bar) 
#                         - gamma_pos * x0 - alpha_pos * v0)
#     x_n = x0 + dt * v_n
#     out_state = jnp.stack([x_n, v_n], -1)
#     out_state = out_state.reshape(-1, self.spatial_dim * self.spatial_dim * self.n_components * self.n_state_channels)
#     return out_state, out_state

#   def initial_state(self, batch_size: Optional[int]):
#     state = jnp.zeros([self.spatial_dim * self.spatial_dim * self.n_components * self.n_state_channels])
#     if batch_size is not None:
#       state = add_batch(state, batch_size)
#     return state



# 1D CoRNN
# class CoRNN(hk.RNNCore):
#   """Couple Oscillator RNN with spatial latent state and locally connected layers
#   """
#   def __init__(
#     self,
#     # spatial_dim: int,
#     spatial_height: int,
#     spatial_width: int,
#     n_components: int,
#     n_state_channels: int,
#     n_input_channels: int,
#     kernel_shape: int,
#     sigma: Activation,
#     gamma_init: float,
#     alpha_init: float,
#     b_bar_init: float,
#     dt_init: float,
#     name: Optional[str] = None
#   ):
#     super().__init__(name=name)
#     self.spatial_height = spatial_height
#     self.spatial_width = spatial_width
#     self.n_components = n_components
#     self.n_state_channels = n_state_channels
#     self.n_input_channels = n_input_channels
#     pad = kernel_shape // 2
#     self.pad_shape = ((0, 0), (0, 0), (pad, pad), (0, 0))
#     self.gamma_init = lambda s, dt: jnp.full(s, gamma_init, dt)
#     self.alpha_init = lambda s, dt: jnp.full(s, alpha_init, dt)
#     self.dt_init = lambda s, dt: jnp.full(s, dt_init, dt)
#     self.b_bar_init = lambda s, dt: jnp.full(s, b_bar_init, dt)
#     if sigma == 'identity':
#       self.sigma = lambda x: x
#     else:
#       self.sigma = utils.get_activation(sigma)

#     # constant_k = 1.0/9.0 # jnp.array([[0,0,0],[0,1,0],[0,0,0]]).reshape((3,3,1,1))
#     ones_state = jnp.ones((1, kernel_shape, self.n_state_channels, self.n_state_channels)) * 1.0/9.0
#     # ones_state_indep = jnp.ones((kernel_shape, kernel_shape, 1, self.n_state_channels)) * 1.0/9.0
#     ones_input = jnp.ones((1, kernel_shape, self.n_input_channels, self.n_state_channels)) * 1.0/9.0
#     constant_k = jnp.array([[0.0, 0.0, 1.0]]).reshape((1, 3, 1, 1))

#     # ones_state = jnp.ones((self.spatial_dim, self.spatial_dim, self.n_state_channels * kernel_shape ** 2, self.n_state_channels)) * 1.0/9.0
#     # ones_input = jnp.ones((self.spatial_dim, self.spatial_dim, self.n_input_channels * kernel_shape ** 2, self.n_state_channels)) * 1.0/9.0
#     coupling_init = lambda s, dt: jnp.full(s, ones_state, dt) + hk.initializers.TruncatedNormal(stddev=0.1)(s, dt)
#     wbar_init = lambda s, dt: jnp.full(s, ones_state, dt) + hk.initializers.TruncatedNormal(stddev=0.1)(s, dt)
#     v_init = lambda s, dt: jnp.full(s, ones_input, dt) + hk.initializers.TruncatedNormal(stddev=0.1)(s, dt)

#     self.coupling = hk.Conv2D(output_channels=self.n_state_channels,
#                               kernel_shape=(1,kernel_shape),
#                               stride=1,
#                               padding='VALID',
#                               # w_init=hk.initializers.Constant(constant_k),
#                               w_init=coupling_init,
#                               # feature_group_count=self.n_state_channels, ## Independant Channels
#                               # mask = constant_k,
#                               with_bias=False,
#                               data_format='NHWC',
#                               name="CouplingNet")

#     self.W_bar = hk.Conv2D(output_channels=self.n_state_channels,
#                             kernel_shape=(1,kernel_shape),
#                             stride=1,
#                             padding='VALID',
#                             # w_init=hk.initializers.Constant(constant_k),
#                             w_init=wbar_init,
#                             # feature_group_count=self.n_state_channels, ## Independant Channels
#                             # mask = constant_k,
#                             with_bias=False,
#                             data_format='NHWC',
#                             name="DampingCouplingNet")

#     self.V = hk.Conv2D(output_channels=self.n_state_channels,
#                         kernel_shape=(1,kernel_shape),
#                         stride=1,
#                         padding='VALID',
#                         # w_init=hk.initializers.Constant(constant_k),
#                         # mask = constant_k
#                         w_init=v_init,
#                         feature_group_count=1,
#                         with_bias=False,
#                         data_format='NHWC',
#                         name="ForcingCouplingNet")

#   def __call__(self, inputs, prev_state):
#     gamma_pos = jax.nn.relu(hk.get_parameter("gamma", shape=[], init=self.gamma_init))
#     alpha_pos = jax.nn.relu(hk.get_parameter("alpha", shape=[], init=self.alpha_init))
#     b_bar = hk.get_parameter("b_bar", shape=[self.spatial_height, self.spatial_width, self.n_state_channels], init=self.b_bar_init)
#     dt = jax.nn.sigmoid(hk.get_parameter("dt", shape=[], init=self.dt_init))

#     # spatial_inputs = inputs.reshape(-1, self.spatial_dim, self.spatial_dim, self.n_components)
#     spatial_inputs = inputs.reshape(-1, self.spatial_height, self.spatial_width, self.n_input_channels)
#     spatial_state = prev_state.reshape(-1, self.spatial_height, self.spatial_width, self.n_state_channels, self.n_components)
#     x0 = spatial_state[:, :, :, :, 0]
#     v0 = spatial_state[:, :, :, :, 1]
#     x_pad = jnp.pad(x0, pad_width=self.pad_shape, mode='wrap')
#     v_pad = jnp.pad(v0, pad_width=self.pad_shape, mode='wrap')
#     u_pad = jnp.pad(spatial_inputs, pad_width=self.pad_shape, mode='wrap')

#     v_n = v0 + dt * (self.sigma(self.coupling(x_pad) 
#                                   + self.W_bar(v_pad)
#                                   + self.V(u_pad)
#                                   + b_bar) 
#                         - gamma_pos * x0 - alpha_pos * v0)
#     x_n = x0 + dt * v_n
#     out_state = jnp.stack([x_n, v_n], -1)
#     out_state = out_state.reshape(-1, self.spatial_height * self.spatial_width * self.n_components * self.n_state_channels)
#     return out_state, out_state

#   def initial_state(self, batch_size: Optional[int]):
#     state = jnp.zeros([self.spatial_height * self.spatial_width * self.n_components * self.n_state_channels])
#     if batch_size is not None:
#       state = add_batch(state, batch_size)
#     return state




def make_flexible_net(
    net_type: str,
    output_dims: int,
    conv_channels: Union[Sequence[int], int],
    num_units: Union[Sequence[int], int],
    num_layers: Optional[int],
    activation: Activation,
    activate_final: bool = False,
    kernel_shapes: Union[Sequence[int], int] = 3,
    strides: Union[Sequence[int], int] = 1,
    padding: Union[Sequence[str], str] = "SAME",
    name: Optional[str] = None,
    **unused_kwargs: Mapping[str, Any]
):
  """Commonly used for creating a flexible network."""
  if unused_kwargs:
    logging.warning("Unused kwargs of `make_flexible_net`: %s",
                    str(unused_kwargs))
  if net_type == "mlp":
    if isinstance(num_units, int):
      assert num_layers is not None
      num_units = [num_units] * (num_layers - 1) + [output_dims]
    else:
      num_units = list(num_units) + [output_dims]
    return DenseNet(
        num_units=num_units,
        activation=activation,
        activate_final=activate_final,
        name=name
    )
  elif net_type == "conv":
    if isinstance(conv_channels, int):
      assert num_layers is not None
      conv_channels = [conv_channels] * (num_layers - 1) + [output_dims]
    else:
      conv_channels = list(conv_channels) + [output_dims]
    return Conv2DNet(
        output_channels=conv_channels,
        kernel_shapes=kernel_shapes,
        strides=strides,
        padding=padding,
        activation=activation,
        activate_final=activate_final,
        name=name
    )
  elif net_type == "transformer":
    raise NotImplementedError()
  else:
    raise ValueError(f"Unrecognized net_type={net_type}.")


def make_flexible_recurrent_net(
    core_type: str,
    net_type: str,
    output_dims: int,
    activate_final: bool = False,
    name: Optional[str] = None,
    net_kwargs: Optional[Mapping[str, Any]] = dict(),
    **latent_system_kwargs,
):
  """Commonly used for creating a flexible recurrences."""
  if net_type != "mlp":
    raise ValueError("We do not support convolutional recurrent nets atm.")
  if latent_system_kwargs:
    logging.warning("Extra kwargs of `make_flexible_recurrent_net` (unused except for CoRNN): %s",
                    str(latent_system_kwargs))
  num_units = net_kwargs['num_units']
  num_layers = net_kwargs['num_layers']

  if isinstance(num_units, (list, tuple)):
    num_units = list(num_units) + [output_dims]
    num_layers = len(num_units)
  else:
    assert num_layers is not None
    num_units = [num_units] * (num_layers - 1) + [output_dims]
  name = name or f"{core_type.upper()}"

  activation = utils.get_activation(net_kwargs['activation'])
  core_list = []
  for i, n in enumerate(num_units):
    if core_type.lower() == "vanilla":
      core_list.append(hk.VanillaRNN(hidden_size=n, name=f"{name}_{i}"))
    elif core_type.lower() == "lstm":
      core_list.append(hk.LSTM(hidden_size=n, name=f"{name}_{i}"))
    elif core_type.lower() == "gru":
      core_list.append(hk.GRU(hidden_size=n, name=f"{name}_{i}"))
    elif core_type.lower() == "cornn":
      core_list.append(CoRNN(**latent_system_kwargs, name=f"{name}_{i}"))
    else:
      raise ValueError(f"Unrecognized core_type={core_type}.")
    if i != num_layers - 1:
      core_list.append(activation)
  if activate_final:
    core_list.append(activation)

  return hk.DeepRNN(core_list, name="RNN")



# def make_coupling_net(
#     net_type: str,
#     kernel_shapes: Union[Sequence[int], int] = 3,
#     strides: Union[Sequence[int], int] = 1,
#     padding: Union[Sequence[str], str] = "SAME",
#     name: Optional[str] = None
# ):
#   """Create Coupling Network (GNN) for Coupled Oscillator Network"""


#     return Conv2DNet(
#         output_channels=conv_channels,
#         kernel_shapes=kernel_shapes,
#         strides=strides,
#         padding=padding,
#         activation=activation,
#         activate_final=activate_final,
#         name=name
#     )