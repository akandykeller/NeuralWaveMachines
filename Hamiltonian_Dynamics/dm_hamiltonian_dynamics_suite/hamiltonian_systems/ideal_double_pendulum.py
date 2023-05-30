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
"""Ideal double pendulum."""
import functools
from typing import Any, Optional, Tuple

from dm_hamiltonian_dynamics_suite.hamiltonian_systems import hamiltonian
from dm_hamiltonian_dynamics_suite.hamiltonian_systems import phase_space
from dm_hamiltonian_dynamics_suite.hamiltonian_systems import utils

import jax
import jax.numpy as jnp


class IdealDoublePendulum(hamiltonian.TimeIndependentHamiltonianSystem):
  """An idealized double pendulum system.

  Parameters:
    m_range - possible range of the particle mass
    g_range - possible range of the gravitational force
    l_range - possible range of the length of the pendulum

  The Hamiltonian is:
      H = [m_2 * l_2^2 * p_1^2 + (m_1 + m_2) * l_1^2 * p_2^2 - 2 * m_2 * l_1 *
      l_2 * p_1 * p_2 * cos(q_1 - q_2)] /
      [2 * m_2 * l_1^2 * l_2^2 * (m_1 + m_2 * sin(q_1 - q_2)^2]
      - (m_1 + m_2) * g * l_1 * cos(q_1) - m_2 * g * l_2 * cos(q_2)

  See https://iopscience.iop.org/article/10.1088/1742-6596/739/1/012066/meta

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
    super().__init__(system_dims=2, **kwargs)
    self.m_range = m_range
    self.g_range = g_range
    self.l_range = l_range
    self.radius_range = radius_range
    self.uniform_annulus = uniform_annulus
    render = functools.partial(
        utils.render_particles_trajectory,
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
    assert len(params) == 5
    m_1 = params["m_1"]
    l_1 = params["l_1"]
    m_2 = params["m_2"]
    l_2 = params["l_2"]
    g = params["g"]

    q_1, q_2 = y.q[..., 0], y.q[..., 1]
    p_1, p_2 = y.p[..., 0], y.p[..., 1]

    a_1 = m_2 * l_2 ** 2 * p_1 ** 2
    a_2 = (m_1 + m_2) * l_1 ** 2 * p_2 ** 2
    a_3 = 2 * m_2 * l_1 * l_2 * p_1 * p_2 * jnp.cos(q_1 - q_2)
    b_1 = 2 * m_2 * l_1 ** 2 * l_2 ** 2
    b_2 = (m_1 + m_2 * jnp.sin(q_1 - q_2) ** 2)
    c_1 = (m_1 + m_2) * g * l_1 * jnp.cos(q_1)
    c_2 = m_2 * g * l_2 * jnp.cos(q_2)
    return (a_1 + a_2 - a_3) / (b_1 * b_2) - c_1 - c_2

  def sample_y(
      self,
      num_samples: int,
      params: utils.Params,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> phase_space.PhaseSpace:
    key1, key2 = jax.random.split(rng_key)
    state_1 = utils.uniform_annulus(
        key1, num_samples, 2, self.radius_range, self.uniform_annulus)
    state_2 = utils.uniform_annulus(
        key2, num_samples, 2, self.radius_range, self.uniform_annulus)
    state = jnp.stack([state_1[..., 0], state_2[..., 0],
                       state_1[..., 1], state_2[..., 1]], axis=-1)
    return phase_space.PhaseSpace.from_state(state.astype(self.dtype))

  def sample_params(
      self,
      num_samples: int,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> utils.Params:
    keys = jax.random.split(rng_key, 5)
    m_1 = jax.random.uniform(keys[0], [num_samples], minval=self.m_range.min,
                             maxval=self.m_range.max)
    m_2 = jax.random.uniform(keys[1], [num_samples], minval=self.m_range.min,
                             maxval=self.m_range.max)
    l_1 = jax.random.uniform(keys[2], [num_samples], minval=self.l_range.min,
                             maxval=self.l_range.max)
    l_2 = jax.random.uniform(keys[3], [num_samples], minval=self.l_range.min,
                             maxval=self.l_range.max)
    g = jax.random.uniform(keys[4], [num_samples], minval=self.g_range.min,
                           maxval=self.g_range.max)
    return dict(m_1=m_1, m_2=m_2, l_1=l_1, l_2=l_2, g=g)

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
    max_d = 2 * self.l_range.max + jnp.sqrt(self.m_range.max / jnp.pi)
    return utils.BoxRegion(-max_d, max_d)

  def canvas_position(
      self,
      position: jnp.ndarray,
      params: utils.Params
  ) -> jnp.ndarray:
    l_1 = utils.expand_to_rank_right(params["l_1"], 2)
    l_2 = utils.expand_to_rank_right(params["l_2"], 2)
    y_1 = jnp.sin(position[..., 0] - jnp.pi / 2.0) * l_1
    x_1 = jnp.cos(position[..., 0] - jnp.pi / 2.0) * l_1
    position_1 = jnp.stack([x_1, y_1], axis=-1)
    y_2 = jnp.sin(position[..., 1] - jnp.pi / 2.0) * l_2
    x_2 = jnp.cos(position[..., 1] - jnp.pi / 2.0) * l_2
    position_2 = jnp.stack([x_2, y_2], axis=-1)
    return jnp.stack([position_1, position_2], axis=-2)

  def render_trajectories(
      self,
      position: jnp.ndarray,
      params: utils.Params,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> Tuple[jnp.ndarray, utils.Params]:
    n, _, d = position.shape
    assert d == self.system_dims
    assert len(params) == 5
    key1, key2 = jax.random.split(rng_key, 2)
    m_1 = params["m_1"]
    m_2 = params["m_2"]
    position = self.canvas_position(position, params)
    position_1, position_2 = position[..., 0, :], position[..., 1, :]
    if self.randomize_canvas_location:
      offset = jax.random.uniform(key1, shape=[n, 2])
      offset = self.random_offset_bounds().convert_from_unit_interval(offset)
    else:
      offset = jnp.zeros([n, 2])
    position_1 = position_1 + offset[:, None, :]
    position_2 = position_1 + position_2
    particles = jnp.stack([position_1, position_2], axis=-2)
    radius_1 = jnp.sqrt(m_1 / jnp.pi)
    radius_2 = jnp.sqrt(m_2 / jnp.pi)
    particles_radius = jnp.stack([radius_1, radius_2], axis=-1)
    if self.num_colors == 1:
      color_index = jnp.zeros([n, 2]).astype("int64")
    else:
      color_index = utils.random_int_k_from_n(
          key2,
          num_samples=n,
          n=self.num_colors,
          k=2
      )
    images = self._batch_render(particles, particles_radius, color_index)
    return images, dict(offset=offset, color_index=color_index)
