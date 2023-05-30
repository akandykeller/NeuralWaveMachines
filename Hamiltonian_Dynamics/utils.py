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
"""Utilities functions for Jax."""
import collections
import functools
from typing import Any, Callable, Dict, Mapping, Union
import math

import distrax
import jax
from jax import core
from jax import lax
from jax import nn
import jax.numpy as jnp
from jax.tree_util import register_pytree_node
from jaxline import utils
import numpy as np
import jax_cfd.base as cfd
import matplotlib as plt

HaikuParams = Mapping[str, Mapping[str, jnp.ndarray]]
Params = Union[Mapping[str, jnp.ndarray], HaikuParams, jnp.ndarray]
_Activation = Callable[[jnp.ndarray], jnp.ndarray]

tf_leaky_relu = functools.partial(nn.leaky_relu, negative_slope=0.2)


def filter_only_scalar_stats(stats):
  return {k: v for k, v in stats.items() if v.size == 1}


def to_numpy(obj):
  return jax.tree_map(np.array, obj)


def normalize_int(x):
  x -= x.min()
  x *= 255 / x.max()
  return x.astype('int')

def hilbert_transform(x, axis=1):
  assert axis == 1 or axis == 0
  N = x.shape[axis]
  # take forward Fourier transform
  s = jnp.fft.fft(x, axis=axis)
  s_anal = jnp.zeros_like(s)
  # double fft energy except @ DC0
  if N > 1:
    if axis == 1:
      s_anal = s_anal.at[:, 1:N//2].set(2 * s[:, 1:N//2])
    if axis == 0:
      s_anal = s_anal.at[1:N//2].set(2 * s[1:N//2])
  else:
    s_anal = s
  x_anal = jnp.fft.ifft(s_anal, axis=axis)
  # take inverse Fourier transform
  return x_anal

# def hilbert_from_scratch(u):
    # N : fft length
    # M : number of elements to zero out
    # U : DFT of u
    # v : IDFT of H(U)

    # N = len(u)
    # # take forward Fourier transform
    # U = fft(u)
    # M = N - N//2 - 1
    # # zero out negative frequency components
    # U[N//2+1:] = [0] * M
    # # double fft energy except @ DC0
    # U[1:N//2] = 2 * U[1:N//2]
    # # take inverse Fourier transform
    # v = ifft(U)
    # return v


@jax.custom_gradient
def geco_lagrange_product(lagrange_multiplier, constraint_ema, constraint_t):
  """Modifies the gradients so that they work as described in GECO.

  The evaluation gives:
    lagrange * C_ema
  The gradient w.r.t lagrange:
    - g * C_t
  The gradient w.r.t constraint_ema:
    0.0
  The gradient w.r.t constraint_t:
    g * lagrange

  Note that if you pass the same value for `constraint_ema` and `constraint_t`
  this would only flip the gradient for the lagrange multiplier.

  Args:
    lagrange_multiplier: The lagrange multiplier
    constraint_ema: The moving average of the constraint
    constraint_t: The current constraint

  Returns:

  """
  def grad(gradient):
    return (- gradient * constraint_t,
            jnp.zeros_like(constraint_ema),
            gradient * lagrange_multiplier)
  return lagrange_multiplier * constraint_ema, grad


def bcast_if(x, t, n):
  return [x] * n if isinstance(x, t) else x


def stack_time_into_channels(
    images: jnp.ndarray,
    data_format: str
) -> jnp.ndarray:
  axis = data_format.index("C")
  list_of_time = [jnp.squeeze(v, axis=1) for v in
                  jnp.split(images, images.shape[1], axis=1)]
  return jnp.concatenate(list_of_time, axis)


def stack_device_dim_into_batch(obj):
  return jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), obj)


def nearest_neighbour_upsampling(x, scale, data_format="NHWC"):
  """Performs nearest-neighbour upsampling."""

  if data_format == "NCHW":
    b, c, h, w = x.shape
    x = jnp.reshape(x, [b, c, h, 1, w, 1])
    ones = jnp.ones([1, 1, 1, scale, 1, scale], dtype=x.dtype)
    return jnp.reshape(x * ones, [b, c, scale * h, scale * w])
  elif data_format == "NHWC":
    b, h, w, c = x.shape
    x = jnp.reshape(x, [b, h, 1, w, 1, c])
    ones = jnp.ones([1, 1, scale, 1, scale, 1], dtype=x.dtype)
    return jnp.reshape(x * ones, [b, scale * h, scale * w, c])
  else:
    raise ValueError(f"Unrecognized data_format={data_format}.")


def get_activation(arg: Union[_Activation, str]) -> _Activation:
  """Returns an activation from provided string."""
  if isinstance(arg, str):
    # Try fetch in order - [this module, jax.nn, jax.numpy]
    if arg in globals():
      return globals()[arg]
    if hasattr(nn, arg):
      return getattr(nn, arg)
    elif hasattr(jnp, arg):
      return getattr(jnp, arg)
    else:
      raise ValueError(f"Unrecognized activation with name {arg}.")
  if not callable(arg):
    raise ValueError(f"Expected a callable, but got {type(arg)}")
  return arg


def merge_first_dims(x: jnp.ndarray, num_dims_to_merge: int = 2) -> jnp.ndarray:
  return x.reshape((-1,) + x.shape[num_dims_to_merge:])


def extract_image(
    inputs: Union[jnp.ndarray, Mapping[str, jnp.ndarray]]
) -> jnp.ndarray:
  """Extracts a tensor with key `image` or `x_image` if it is a dict, otherwise returns the inputs."""
  if isinstance(inputs, dict):
    if "image" in inputs:
      return inputs["image"]
    else:
      return inputs["x_image"]
  elif isinstance(inputs, jnp.ndarray):
    return inputs
  raise NotImplementedError(f"Not implemented of inputs of type"
                            f" {type(inputs)}.")


def extract_gt_state(inputs: Any) -> jnp.ndarray:
  if isinstance(inputs, dict):
    return inputs["x"]
  elif not isinstance(inputs, jnp.ndarray):
    raise NotImplementedError(f"Not implemented of inputs of type"
                              f" {type(inputs)}.")
  return inputs


def reshape_latents_conv_to_flat(conv_latents, axis_n_to_keep=1):
  q, p = jnp.split(conv_latents, 2, axis=-1)
  q = jax.tree_map(lambda x: x.reshape(x.shape[:axis_n_to_keep] + (-1,)), q)
  p = jax.tree_map(lambda x: x.reshape(x.shape[:axis_n_to_keep] + (-1,)), p)
  flat_latents = jnp.concatenate([q, p], axis=-1)

  return flat_latents


def triu_matrix_from_v(x, ndim):
  assert x.shape[-1] == (ndim * (ndim + 1)) // 2
  matrix = jnp.zeros(x.shape[:-1] + (ndim, ndim))
  idx = jnp.triu_indices(ndim)
  index_update = lambda x, idx, y: x.at[idx].set(y)
  for _ in range(x.ndim - 1):
    index_update = jax.vmap(index_update, in_axes=(0, None, 0))
  return index_update(matrix, idx, x)


def flatten_dict(d, parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
  items = []
  for k, v in d.items():
    new_key = parent_key + sep + k if parent_key else k
    if isinstance(v, collections.MutableMapping):
      items.extend(flatten_dict(v, new_key, sep=sep).items())
    else:
      items.append((new_key, v))
  return dict(items)


def convert_to_pytype(target, reference):
  """Makes target the same pytype as reference, by jax.tree_flatten."""
  _, pytree = jax.tree_flatten(reference)
  leaves, _ = jax.tree_flatten(target)
  return jax.tree_unflatten(pytree, leaves)


def func_if_not_scalar(func):
  """Makes a function that uses func only on non-scalar values."""
  @functools.wraps(func)
  def wrapped(array, axis=0):
    if array.ndim == 0:
      return array
    return func(array, axis=axis)
  return wrapped


mean_if_not_scalar = func_if_not_scalar(jnp.mean)


class MultiBatchAccumulator(object):
  """Class for abstracting statistics accumulation over multiple batches."""

  def __init__(self):
    self._obj = None
    self._obj_max = None
    self._obj_min = None
    self._num_samples = None

  def add(self, averaged_values, num_samples):
    """Adds an element to the moving average and the max."""
    if self._obj is None:
      self._obj_max = jax.tree_map(lambda y: y * 1.0, averaged_values)
      self._obj_min = jax.tree_map(lambda y: y * 1.0, averaged_values)
      self._obj = jax.tree_map(lambda y: y * num_samples, averaged_values)
      self._num_samples = num_samples
    else:
      self._obj_max = jax.tree_multimap(jnp.maximum, self._obj_max,
                                        averaged_values)
      self._obj_min = jax.tree_multimap(jnp.minimum, self._obj_min,
                                        averaged_values)
      self._obj = jax.tree_multimap(lambda x, y: x + y * num_samples, self._obj,
                                    averaged_values)
      self._num_samples += num_samples

  def value(self):
    return jax.tree_map(lambda x: x / self._num_samples, self._obj)

  def max(self):
    return jax.tree_map(float, self._obj_max)

  def min(self):
    return jax.tree_map(float, self._obj_min)

  def sum(self):
    return self._obj


register_pytree_node(
    distrax.Normal,
    lambda instance: ([instance.loc, instance.scale], None),
    lambda _, args: distrax.Normal(*args)
)


def inner_product(x: Any, y: Any) -> jnp.ndarray:
  products = jax.tree_multimap(lambda x_, y_: jnp.sum(x_ * y_), x, y)
  return sum(jax.tree_leaves(products))


get_first = utils.get_first
bcast_local_devices = utils.bcast_local_devices
py_prefetch = utils.py_prefetch
p_split = jax.pmap(lambda x, num: list(jax.random.split(x, num)),
                   static_broadcasted_argnums=1)


def wrap_if_pmap(p_func):
  def p_func_if_pmap(obj, axis_name):
    try:
      core.axis_frame(axis_name)
      return p_func(obj, axis_name)
    except NameError:
      return obj
  return p_func_if_pmap


pmean_if_pmap = wrap_if_pmap(lax.pmean)
psum_if_pmap = wrap_if_pmap(lax.psum)


def grad_vec(x, axis=[2,3]):
    x_pad = jnp.pad(x, ((0,0),(0,0),(1,1),(1,1)), mode='wrap')
    return jnp.stack(jnp.gradient(x_pad, axis=axis), dim=0)[:, :, :, 1:-1, 1:-1]

def grad_scalar(x, axis=[2,3]):
    x_pad = jnp.pad(x, ((0,0),(0,0),(1,1),(1,1)), mode='wrap')
    return jnp.gradient(x_pad, axis=axis)[:, :, 1:-1, 1:-1]

def div_vec(x, axis=[2,3]):
    x_pad = jnp.pad(x, ((0,0),(0,0),(1,1),(1,1)), mode='wrap')
    grads = jnp.gradient(x_pad, axis=axis)
    dx_dx = grads[0][:, :, 1:-1, 1:-1]
    dy_dy = grads[1][:, :, 1:-1, 1:-1]
    return dx_dx + dy_dy

def laplace_vec(x, axis=[2,3]):
    res = jnp.zeros_like(x)
    for a in axis:
        x_pad = jnp.pad(x, ((0,0),(0,0),(1,1),(1,1)), mode='wrap')
        del1 = jnp.gradient(x_pad, axis=a)[0][:, :, 1:-1, 1:-1]

        del1_pad = jnp.pad(del1, ((0,0),(0,0),(1,1),(1,1)), mode='wrap')
        del2 = jnp.gradient(del1_pad, axis=a)[0][:, :, 1:-1, 1:-1]
        res += del2
    return res

def laplace_scalar(x, axis=[2,3]):
    res = jnp.zeros_like(x)
    for a in axis:        
        del1 = grad_scalar(x, axis=a)
        del2 = grad_scalar(del1, axis=a)
        res += del2
    return res

def laplace_scalar_singlepad(x, axis=[2,3]):
    res = jnp.zeros_like(x)
    for a in axis:
        x_pad = jnp.pad(x, ((0,0),(0,0),(1,1),(1,1)), mode='wrap')
        del1 = jnp.gradient(x_pad, axis=a)
        del2 = jnp.gradient(del1, axis=a)[:, :, 1:-1, 1:-1]
        res += del2
    return res

def phase_divergence_loss(x, spatial_shape, axis=1):
  assert axis == 1
  x_anal = hilbert_transform(x)
  phase = jnp.angle(x_anal)

  # B, T, H, W, C
  assert spatial_shape[-1] == 1
  
  spatial_phase = phase.reshape(phase.shape[0], *spatial_shape)[:,:,:,:,0]

  return laplace_scalar_singlepad(spatial_phase, axis=(2,3))


def phase_roll(x_anal, dt=0.1):
  x_anal_tp1 = x_anal * jnp.exp(1j * dt) * jnp.abs(x_anal)
  x_real_tp1 = jnp.real(x_anal_tp1)
  return x_anal_tp1, x_real_tp1
  
def get_anal_init(x_real):
    x_anal = hilbert_transform(x_real, axis=0)
    return x_anal[0]

def phase_velocity(x_real):
  x_anal = hilbert_transform(x_real, axis=0)
  # phase = jnp.angle(x_anal)
  
  x_anal_xp1 = jnp.roll(x_anal, shift=1, axis=2)
  x_anal_xm1 = jnp.roll(x_anal, shift=-1, axis=2)
  x_anal_yp1 = jnp.roll(x_anal, shift=1, axis=3)
  x_anal_ym1 = jnp.roll(x_anal, shift=-1, axis=3)

  dx = jnp.angle(x_anal_xp1 * jnp.conj(x_anal_xm1)) / 2.0
  dy = jnp.angle(x_anal_yp1 * jnp.conj(x_anal_ym1)) / 2.0

  vel_mag = (dx ** 2.0 + dy ** 2.0) ** 0.5
  return vel_mag

def percent_waves(x_real, rng_key):
  rng, key = p_split(rng_key, 2)
  vel = phase_velocity(x_real)
  lambda_x = 1.0 / jnp.abs(vel)

  x_shuff = jax.random.shuffle(key[0], x_real, -1)
  vel_shuff = phase_velocity(x_shuff)
  lambda_shuff = 1.0 / jnp.abs(vel_shuff)
  cutoff = jnp.quantile(lambda_shuff, 0.99)
  pct_waves = jnp.sum(lambda_x > cutoff) / float(jnp.size(lambda_x))
  return pct_waves

def rgb_phase(x_real):
  x_anal = hilbert_transform(x_real, axis=0)
  phase = jnp.angle(x_anal)
  phase_r = jnp.cos(phase)
  phase_g = jnp.cos(phase + 2.0944) 
  phase_b = jnp.cos(phase - 2.0944) 
  rgb = jnp.concatenate([phase_r, phase_g, phase_b], axis=1)
  return rgb

# def laplace_cfd(x, grid, bcs):
#   x_gv = cfd.initial_conditions.wrap_velocities([x], grid, [bcs])[0]
#   return cfd.finite_differences.laplacian(x_gv).data

# w1 = jnp.zeros((1,1,k,k))
# w1[:,:,0:k//2, k//2] = -1.0/(k-1)
# w1[:,:,k//2+1:, k//2] = 1.0/(k-1)

# w2 = jnp.zeros((1,1,k,k))
# w2[:,:,k//2,0:k//2] = -1.0/(k-1)
# w2[:,:,k//2,k//2+1:] = 1.0/(k-1)

# def grad_conv(x, axis=[1,2], pad=k//2):
#     """
#     Gradient implemented with convolution by fixed kernel
#     """
#     x_pad = torch.Tensor(np.pad(x, ((0,0), (0,0),(pad,pad),(pad,pad)), mode='wrap'))
#     return F.conv2d(x_pad, w, bias=None).numpy()

# def laplace_conv(x, axis=[1,2], ws=None, k=3):    
#     res = jnp.zeros_like(x)
#     for a in axis:
#         del1 = grad_conv(x, ws, axis=a)
#         del2 = grad_conv(del1, ws, axis=a)
#         res += del2
#     return res

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = jnp.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).reshape(kernel_size, kernel_size)
    y_grid = x_grid.T
    xy_grid = jnp.stack([x_grid, y_grid], axis=-1).astype('float')

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      jnp.exp(
                          -jnp.sum((xy_grid - mean)**2., -1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / jnp.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.reshape(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, axis=0)
    
    return gaussian_kernel


def get_gaussian_filter(kernel_size=3, sigma=2, channels=1):
    gaussian_kernel = get_gaussian_kernel(kernel_size=kernel_size, sigma=sigma, channels=channels)
    pad = kernel_size // 2
    def gf(x):
      x_pad = jnp.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='wrap')
      return jax.lax.conv_general_dilated(x_pad, gaussian_kernel, 
                                              window_strides=[1,1], 
                                              padding='VALID')
    
    return gf


