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
"""Module with various utilities not central to the code."""
from typing import Dict, Optional, Tuple, Union

import jax
from jax import lax
import jax.numpy as jnp
import jax.random as jnr
import numpy as np


FloatArray = Union[float, jnp.ndarray]
Params = Dict[str, jnp.ndarray]


class BoxRegion:
  """A class for bounds, especially used for sampling."""

  def __init__(self, minimum: FloatArray, maximum: FloatArray):
    minimum = jnp.asarray(minimum)
    maximum = jnp.asarray(maximum)
    if minimum.shape != maximum.shape:
      raise ValueError(f"The passed values for minimum and maximum should have "
                       f"the same shape, but had shapes: {minimum.shape} "
                       f"and {maximum.shape}.")
    self._minimum = minimum
    self._maximum = maximum

  @property
  def min(self) -> jnp.ndarray:
    return self._minimum

  @property
  def max(self) -> jnp.ndarray:
    return self._maximum

  @property
  def size(self) -> jnp.ndarray:
    return self.max - self.min

  @property
  def dims(self) -> int:
    if self.min.ndim != 0:
      return self.min.shape[-1]
    return 0

  def convert_to_unit_interval(self, value: jnp.ndarray) -> jnp.ndarray:
    return (value - self.min) / self.size

  def convert_from_unit_interval(self, value: jnp.ndarray) -> jnp.ndarray:
    return value * self.size + self.min

  def __str__(self) -> str:
    return f"{type(self).__name__}(min={self.min}, max={self.max})"

  def __repr__(self) -> str:
    return self.__str__()


def expand_to_rank_right(x: jnp.ndarray, rank: int) -> jnp.ndarray:
  if x.ndim == rank:
    return x
  assert x.ndim < rank
  new_shape = x.shape + (1,) * (rank - x.ndim)
  return x.reshape(new_shape)


def expand_to_rank_left(x: jnp.ndarray, rank: int) -> int:
  if x.ndim == rank:
    return x
  assert x.ndim < rank
  new_shape = (1,) * (rank - x.ndim) + x.shape
  return x.reshape(new_shape)


def vecmul(matrix: jnp.ndarray, vector: jnp.ndarray) -> jnp.ndarray:
  return jnp.matmul(matrix, vector[..., None])[..., 0]


def dt_to_t_eval(t0: FloatArray, dt: FloatArray, num_steps: int) -> jnp.ndarray:
  if (isinstance(t0, (float, np.ndarray)) and
      isinstance(dt, (float, np.ndarray))):
    dt = np.asarray(dt)[None]
    shape = [num_steps] + [1] * (dt.ndim - 1)
    return t0 + dt * np.arange(1, num_steps + 1).reshape(shape)
  else:
    return t0 + dt * jnp.arange(1, num_steps + 1)


def t_eval_to_dt(t0: FloatArray, t_eval: FloatArray) -> jnp.ndarray:
  t = jnp.ones_like(t_eval[:1]) * t0
  t = jnp.concatenate([t, t_eval], axis=0)
  return t[1:] - t[:-1]


def simple_loop(
    f,
    x0: jnp.ndarray,
    t_args: Optional[jnp.ndarray] = None,
    num_steps: Optional[int] = None,
    use_scan: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Runs a simple loop that outputs the evolved variable at every time step."""
  if t_args is None and num_steps is None:
    raise ValueError("Exactly one of `t_args` and `num_steps` should be "
                     "provided.")
  if t_args is not None and num_steps is not None:
    raise ValueError("Exactly one of `t_args` and `num_steps` should be "
                     "provided.")

  def step(x_t, t_arg):
    x_next = f(x_t) if t_arg is None else f(x_t, t_arg)
    return x_next, x_next
  if use_scan:
    return lax.scan(step, init=x0, xs=t_args, length=num_steps)[1]

  y = []
  x = x0
  num_steps = t_args.shape[0] if t_args is not None else num_steps
  t_args = [None] * num_steps if t_args is None else t_args
  for i in range(num_steps):
    x, _ = step(x, t_args[i])
    y.append(x)
  return jax.tree_multimap(lambda *args: jnp.stack(args, axis=0), *y)


def hsv2rgb(array: jnp.ndarray) -> jnp.ndarray:
  """Converts an HSV float array image to RGB."""
  hi = jnp.floor(array[..., 0] * 6)
  f = array[..., 0] * 6 - hi
  p = array[..., 2] * (1 - array[..., 1])
  q = array[..., 2] * (1 - f * array[..., 1])
  t = array[..., 2] * (1 - (1 - f) * array[..., 1])
  v = array[..., 2]

  hi = jnp.stack([hi, hi, hi], axis=-1).astype(jnp.uint8) % 6
  hi_is_0 = hi == 0
  hi_is_1 = hi == 1
  hi_is_2 = hi == 2
  hi_is_3 = hi == 3
  hi_is_4 = hi == 4
  hi_is_5 = hi == 5
  out = (hi_is_0 * jnp.stack((v, t, p), axis=-1) +
         hi_is_1 * jnp.stack((q, v, p), axis=-1) +
         hi_is_2 * jnp.stack((p, v, t), axis=-1) +
         hi_is_3 * jnp.stack((p, q, v), axis=-1) +
         hi_is_4 * jnp.stack((t, p, v), axis=-1) +
         hi_is_5 * jnp.stack((v, p, q), axis=-1))
  return out


def render_particles_trajectory(
    particles: jnp.ndarray,
    particles_radius: FloatArray,
    color_indices: FloatArray,
    canvas_limits: BoxRegion,
    resolution: int,
    num_colors: int,
    background_color: Tuple[float, float, float] = (0.321, 0.349, 0.368),
    temperature: FloatArray = 80.0):
  """Renders n particles in different colors for a full trajectory.

  NB: The default background color is not black as we have experienced issues
    when training models with black background.

  Args:
    particles: Array of size (t, n, 2)
      The last 2 dimensions define the x, y coordinates of each particle.
    particles_radius: Array of size (n,) or a single value.
      Defines the radius of each particle.
    color_indices: Array of size (n,) or a single value.
      Defines the color of each particle.
    canvas_limits: List of 2 lists or Array of size (2, 2)
      First row defines the limit over x and second over y.
    resolution: int
      The resolution of the produced images.
    num_colors: int
      The number of possible colors to use.
    background_color: List or Array of size (3)
      The color for the background. Default to black.
    temperature: float
      The temperature of the sigmoid distance metric used to the center of the
      particles.

  Returns:
    An array of size (t, resolution, resolution, 3) with the produced images.
  """
  particles = jnp.asarray(particles)
  assert particles.ndim == 3
  assert particles.shape[-1] == 2
  t, n = particles.shape[:2]
  particles_radius = jnp.asarray(particles_radius)
  if particles_radius.ndim == 0:
    particles_radius = jnp.full([n], particles_radius)
  assert particles_radius.shape == (n,)
  color_indices = jnp.asarray(color_indices)
  if color_indices.ndim == 0:
    color_indices = jnp.full([n], color_indices)
  assert color_indices.shape == (n,), f"Colors shape: {color_indices.shape}"
  background_color = jnp.asarray(background_color)
  assert background_color.shape == (3,)

  particles = canvas_limits.convert_to_unit_interval(particles)
  canvas_size = canvas_limits.max - canvas_limits.min
  canvas_size = canvas_size[0] if canvas_size.ndim == 1 else canvas_size
  particles_radius = particles_radius / canvas_size
  images = jnp.ones([t, resolution, resolution, 3]) * background_color

  hues = jnp.linspace(0, 1, num=num_colors, endpoint=False)
  colors = hues[color_indices][None, :, None, None]
  s_channel = jnp.ones((t, n, resolution, resolution))
  v_channel = jnp.ones((t, n, resolution, resolution))
  h_channel = jnp.ones((t, n, resolution, resolution)) * colors
  hsv_imgs = jnp.stack((h_channel, s_channel, v_channel), axis=-1)
  rgb_imgs = hsv2rgb(hsv_imgs)
  images = [img[:, 0] for img in jnp.split(rgb_imgs, n, axis=1)] + [images]

  grid = jnp.linspace(0.0, 1.0, resolution)
  dx, dy = jnp.meshgrid(grid, grid)
  dx, dy = dx[None, None], dy[None, None]
  x, y = particles[..., 0][..., None, None], particles[..., 1][..., None, None]
  d = jnp.sqrt((x - dx) ** 2 + (y - dy) ** 2)
  particles_radius = particles_radius[..., None, None]
  mask = 1.0 / (1.0 + jnp.exp((d - particles_radius) * temperature))
  masks = ([m[:, 0, ..., None] for m in jnp.split(mask, n, axis=1)] +
           [jnp.ones_like(images[0])])

  final_image = jnp.zeros([t, resolution, resolution, 3])
  c = jnp.ones_like(images[0])
  for img, m in zip(images, masks):
    final_image = final_image + c * m * img
    c = c * (1 - m)
  return final_image


def uniform_annulus(
    key: jnp.ndarray,
    num_samples: int,
    dim_samples: int,
    radius_range: BoxRegion,
    uniform: bool
) -> jnp.ndarray:
  """Samples points uniformly in the annulus defined by radius range."""
  key1, key2 = jnr.split(key)
  direction = jnr.normal(key1, [num_samples, dim_samples])
  norms = jnp.linalg.norm(direction, axis=-1, keepdims=True)
  direction = direction / norms
  # Sample a radius uniformly between [min_radius, max_radius]
  r = jnr.uniform(key2, [num_samples])
  if uniform:
    radius_range = BoxRegion(radius_range.min ** 2, radius_range.max ** 2)
    r = jnp.sqrt(radius_range.convert_from_unit_interval(r))
  else:
    r = radius_range.convert_from_unit_interval(r)
  return direction * r[:, None]


multi_shuffle = jax.vmap(lambda x, key, k: jnr.permutation(key, x)[:k],
                         in_axes=(0, 0, None), out_axes=0)


def random_int_k_from_n(
    rng: jnp.ndarray,
    num_samples: int,
    n: int,
    k: int
) -> jnp.ndarray:
  """Samples randomly k integers from 1 to n."""
  if k > n:
    raise ValueError(f"k should be less than or equal to n, but got k={k} and "
                     f"n={n}.")
  x = jnp.repeat(jnp.arange(n).reshape([1, n]), num_samples, axis=0)
  return multi_shuffle(x, jnr.split(rng, num_samples), k)
