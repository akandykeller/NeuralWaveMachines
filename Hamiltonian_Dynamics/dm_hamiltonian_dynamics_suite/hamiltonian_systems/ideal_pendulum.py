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
"""Ideal pendulum."""
import functools
from typing import Any, Optional, Tuple

from dm_hamiltonian_dynamics_suite.hamiltonian_systems import hamiltonian
from dm_hamiltonian_dynamics_suite.hamiltonian_systems import phase_space
from dm_hamiltonian_dynamics_suite.hamiltonian_systems import utils

import jax
import jax.numpy as jnp


class IdealPendulum(hamiltonian.TimeIndependentHamiltonianSystem):
  """An idealized pendulum system.

  Parameters:
    m_range - possible range of the particle mass
    g_range - possible range of the gravitational force
    l_range - possible range of the length of the pendulum

  The Hamiltonian is:
      m * l * g * (1 - cos(q)) + p^2 / (2 * m * l^2)

  Initial state parameters:
    radius_range - The initial state is sampled from a disk in phase space with
      radius in this range.
    uniform_annulus - Whether to sample uniformly on the disk or uniformly the
      radius.
    randomize_canvas_location - Whether to randomize th vertical position of the
      particle when rendering.
  """

  def __init__(
      self,
      m_range: utils.BoxRegion,
      g_range: utils.BoxRegion,
      l_range: utils.BoxRegion,
      radius_range: utils.BoxRegion,
      uniform_annulus: bool = True,
      **kwargs):
    super().__init__(system_dims=1, **kwargs)
    self.m_range = m_range
    self.g_range = g_range
    self.l_range = l_range
    self.radius_range = radius_range
    self.uniform_annulus = uniform_annulus
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
    assert len(params) == 3
    m = params["m"]
    l = params["l"]
    g = params["g"]
    potential = m * g * l *  (1 - jnp.cos(y.q[..., 0]))
    kinetic = y.p[..., 0] ** 2 / (2 * m * l ** 2)
    return potential + kinetic

  def sample_y(
      self,
      num_samples: int,
      params: utils.Params,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> phase_space.PhaseSpace:
    state = utils.uniform_annulus(
        rng_key, num_samples, 2, self.radius_range, self.uniform_annulus)
    return phase_space.PhaseSpace.from_state(state.astype(self.dtype))

  def sample_params(
      self,
      num_samples: int,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> utils.Params:
    key1, key2, key3 = jax.random.split(rng_key, 3)
    m = jax.random.uniform(key1, [num_samples], minval=self.m_range.min,
                           maxval=self.m_range.max)
    l = jax.random.uniform(key2, [num_samples], minval=self.l_range.min,
                           maxval=self.l_range.max)
    g = jax.random.uniform(key3, [num_samples], minval=self.g_range.min,
                           maxval=self.g_range.max)
    return dict(m=m, l=l, g=g)

  def simulate_analytically(
      self,
      y0: phase_space.PhaseSpace,
      t0: utils.FloatArray,
      t_eval: jnp.ndarray,
      params: utils.Params,
      **kwargs: Any
  ) -> Optional[phase_space.PhaseSpace]:
    return None

  def canvas_bounds(self) -> utils.BoxRegion:
    max_d = self.l_range.max + jnp.sqrt(self.m_range.max / jnp.pi)
    return utils.BoxRegion(-max_d, max_d)

  def canvas_position(
      self,
      position: jnp.ndarray,
      params: utils.Params
  ) -> jnp.ndarray:
    l = utils.expand_to_rank_right(params["l"], 2)
    y = jnp.sin(position[..., 0] - jnp.pi / 2.0) * l
    x = jnp.cos(position[..., 1] - jnp.pi / 2.0) * l
    return jnp.stack([x, y], axis=-1)

  def render_trajectories(
      self,
      position: jnp.ndarray,
      params: utils.Params,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> Tuple[jnp.ndarray, utils.Params]:
    n, _, d = position.shape
    assert d == self.system_dims
    assert len(params) == 3
    m = utils.expand_to_rank_right(params["m"], 2)
    key1, key2 = jax.random.split(rng_key, 2)
    particles = self.canvas_position(position, params)
    if self.randomize_canvas_location:
      offset = jax.random.uniform(key1, shape=[n, 2])
      offset = self.random_offset_bounds().convert_from_unit_interval(offset)
    else:
      offset = jnp.zeros(shape=[n, 2])
    particles = particles + offset[:, None, :]
    particles = particles[..., None, :]
    particles_radius = jnp.sqrt(m / jnp.pi)
    if self.num_colors == 1:
      color_index = jnp.zeros([n, 1]).astype("int64")
    else:
      color_index = jax.random.randint(
          key=key2, shape=[n, 1], minval=0, maxval=self.num_colors)
    images = self._batch_render(particles, particles_radius, color_index)
    return images, dict(offset=offset, color_index=color_index)
