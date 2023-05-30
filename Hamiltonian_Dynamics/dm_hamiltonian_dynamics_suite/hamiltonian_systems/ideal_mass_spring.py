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
"""Ideal mass spring."""
import functools
from typing import Any, Optional, Tuple

from dm_hamiltonian_dynamics_suite.hamiltonian_systems import hamiltonian
from dm_hamiltonian_dynamics_suite.hamiltonian_systems import phase_space
from dm_hamiltonian_dynamics_suite.hamiltonian_systems import utils

import jax
import jax.numpy as jnp


class IdealMassSpring(hamiltonian.TimeIndependentHamiltonianSystem):
  """An idealized mass-spring system (also known as a harmonica oscillator).

  The system is represented in 2 dimensions, but the spring moves only in the
  vertical orientation.

  Parameters:
      k_range - possible range of the spring's force coefficient
      m_range - possible range of the particle mass

  The Hamiltonian is:
      k * q^2 / 2.0 + p^2 / (2 * m)

  Initial state parameters:
    radius_range - The initial state is sampled from a disk in phase space with
      radius in this range.
    uniform_annulus - Whether to sample uniformly on the disk or uniformly the
      radius.
    randomize_x - Whether to randomize the horizontal position of the particle
      when rendering.
  """

  def __init__(
      self,
      k_range: utils.BoxRegion,
      m_range: utils.BoxRegion,
      radius_range: utils.BoxRegion,
      uniform_annulus: bool = True,
      randomize_x: bool = True,
      **kwargs):
    super().__init__(system_dims=1, **kwargs)
    self.k_range = k_range
    self.m_range = m_range
    self.radius_range = radius_range
    self.uniform_annulus = uniform_annulus
    self.randomize_x = randomize_x
    render = functools.partial(utils.render_particles_trajectory,
                               canvas_limits=self.full_canvas_bounds(),
                               resolution=self.resolution,
                               num_colors=self.num_colors)
    self._batch_render = jax.vmap(render)

  def _hamiltonian(
      self,
      y: phase_space.PhaseSpace,
      params: utils.Params,
      **kwargs: Any
  ) -> jnp.ndarray:
    assert len(params) == 2
    k = params["k"]
    m = params["m"]
    potential = k * y.q[..., 0] ** 2 / 2
    kinetic = y.p[..., 0] ** 2 / (2 * m)
    return potential + kinetic

  def sample_y(
      self,
      num_samples: int,
      params: utils.Params,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> phase_space.PhaseSpace:
    assert len(params) == 2
    k = params["k"]
    m = params["m"]
    state = utils.uniform_annulus(
        rng_key, num_samples, 2, self.radius_range, self.uniform_annulus)
    q = state[..., :1]
    p = state[..., 1:] * jnp.sqrt(k * m)
    return phase_space.PhaseSpace(position=q, momentum=p)

  def sample_params(
      self,
      num_samples: int,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> utils.Params:
    key1, key2 = jax.random.split(rng_key)
    k = jax.random.uniform(key1, [num_samples], minval=self.k_range.min,
                           maxval=self.k_range.max)
    m = jax.random.uniform(key2, [num_samples], minval=self.m_range.min,
                           maxval=self.m_range.max)
    return dict(k=k, m=m)

  def simulate_analytically(
      self,
      y0: phase_space.PhaseSpace,
      t0: utils.FloatArray,
      t_eval: jnp.ndarray,
      params: utils.Params,
      **kwargs: Any
  ) -> Optional[phase_space.PhaseSpace]:
    if self.friction != 0.0:
      return None
    assert len(params) == 2
    k = params["k"]
    m = params["m"]
    t = t_eval - t0
    w = jnp.sqrt(k / m).astype(self.dtype)
    a = jnp.sqrt(y0.q[..., 0] ** 2 + y0.p[..., 0] ** 2 / (k * m))
    b = jnp.arctan2(- y0.p[..., 0], y0.q[..., 0] * m * w)
    w, a, b, m = w[..., None], a[..., None], b[..., None], m[..., None]
    t = utils.expand_to_rank_right(t, y0.q.ndim + 1)

    q = a * jnp.cos(w * t + b)
    p = - a * m * w * jnp.sin(w * t + b)
    return phase_space.PhaseSpace(position=q, momentum=p)

  def canvas_bounds(self) -> utils.BoxRegion:
    max_x = self.radius_range.max
    max_r = jnp.sqrt(self.m_range.max / jnp.pi)
    return utils.BoxRegion(- max_x - max_r, max_x + max_r)

  def canvas_position(
      self,
      position: jnp.ndarray,
      params: utils.Params
  ) -> jnp.ndarray:
    return jnp.stack([jnp.zeros_like(position), position], axis=-1)

  def render_trajectories(
      self,
      position: jnp.ndarray,
      params: utils.Params,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> Tuple[jnp.ndarray, utils.Params]:
    n, _, d = position.shape
    assert d == self.system_dims
    assert len(params) == 2
    key1, key2 = jax.random.split(rng_key)
    m = utils.expand_to_rank_right(params["m"], 2)
    particles = self.canvas_position(position, params)
    if self.randomize_x:
      x_offset = jax.random.uniform(key1, shape=[n])
      y_offset = jnp.zeros_like(x_offset)
      offset = jnp.stack([x_offset, y_offset], axis=-1)
    else:
      offset = jnp.zeros([n, d])
    if self.randomize_canvas_location:
      offset_ = jax.random.uniform(key2, shape=[n, d])
      offset_ = self.random_offset_bounds().convert_from_unit_interval(offset_)
      offset = offset + offset_
    particles = particles + offset[:, None, None, :]
    particles_radius = jnp.sqrt(m / jnp.pi)
    if self.num_colors == 1:
      color_index = jnp.zeros([n, 1]).astype("int64")
    else:
      color_index = jax.random.randint(
          key=rng_key, shape=[n, 1], minval=0, maxval=self.num_colors)
    images = self._batch_render(particles, particles_radius, color_index)
    return images, dict(offset=offset, color_index=color_index)
