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
"""N body."""
import abc
import functools
from typing import Any, Optional, Tuple

from dm_hamiltonian_dynamics_suite.hamiltonian_systems import hamiltonian
from dm_hamiltonian_dynamics_suite.hamiltonian_systems import phase_space
from dm_hamiltonian_dynamics_suite.hamiltonian_systems import utils

import jax
import jax.numpy as jnp


class NBodySystem(hamiltonian.TimeIndependentHamiltonianSystem):
  """An N-body system abstract class.

  Parameters:
    m_range - possible range of the particle mass
    g_range - possible range of the gravitational force
    provided_canvas_bounds - The canvas bounds for the given ranges

  The Hamiltonian is:
      - sum_i<j (g * m_i * m_j) / ||q_i-q_j|| + sum_i ||p_i||^2 / (2 * m_i)

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
      n: int,
      space_dims: int,
      m_range: utils.BoxRegion,
      g_range: utils.BoxRegion,
      provided_canvas_bounds: utils.BoxRegion,
      **kwargs):
    super(NBodySystem, self).__init__(
        system_dims=n * space_dims,
        **kwargs,
    )
    self.n = n
    self.space_dims = space_dims
    self.m_range = m_range
    self.g_range = g_range
    self.provided_canvas_bounds = provided_canvas_bounds
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
    m = params["m"]
    g = params["g"]
    q = y.q.reshape([-1, self.n, self.space_dims])
    p = y.p.reshape([-1, self.n, self.space_dims])

    q_ij = jnp.matmul(q, jnp.swapaxes(q, axis1=-1, axis2=-2))
    q_ii = jnp.diagonal(q_ij, axis1=-1, axis2=-2)
    q_ij_norms_2 = q_ii[:, None, :] + q_ii[:, :, None] - 2.0 * q_ij
    # Adding identity so that on the diagonal the norms are not 0
    q_ij_norms = jnp.sqrt(q_ij_norms_2 + jnp.identity(self.n))
    masses_ij = m[:, None, :] * m[:, :, None]
    # Remove masses in the diagonal so that those potentials are 0
    masses_ij = masses_ij - masses_ij * jnp.identity(self.n)[None]
    # Compute pairwise interactions
    products = g[:, None, None] * masses_ij / q_ij_norms
    # Note that here we are summing both i->j and j->i hence the division by 2
    potential = - products.sum(axis=(-2, -1)) / 2
    kinetic = jnp.sum(p ** 2, axis=-1) / (2.0 * m)
    kinetic = kinetic.sum(axis=-1)
    return potential + kinetic

  @abc.abstractmethod
  def sample_y(
      self,
      num_samples: int,
      params: utils.Params,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> phase_space.PhaseSpace:
    pass

  def sample_params(
      self,
      num_samples: int,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> utils.Params:
    key1, key2 = jax.random.split(rng_key)
    m = jax.random.uniform(key1, [num_samples, self.n], minval=self.m_range.min,
                           maxval=self.m_range.max)
    g = jax.random.uniform(key2, [num_samples], minval=self.g_range.min,
                           maxval=self.g_range.max)
    return dict(m=m, g=g)

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
    return self.provided_canvas_bounds

  def canvas_position(
      self,
      position: jnp.ndarray,
      params: utils.Params
  ) -> jnp.ndarray:
    return position

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
    key1, key2 = jax.random.split(rng_key, 2)
    m = utils.expand_to_rank_right(params["m"], 2)
    particles = position.reshape([n, -1, self.n, self.space_dims])
    if self.randomize_canvas_location:
      offset = jax.random.uniform(key1, shape=[n, self.space_dims])
      offset = self.random_offset_bounds().convert_from_unit_interval(offset)
    else:
      offset = jnp.zeros(shape=[n, self.space_dims])
    particles = particles + offset[:, None, None, :]
    particles_radius = jnp.sqrt(m / jnp.pi)
    if self.num_colors == 1:
      color_index = jnp.zeros([n, self.n]).astype("int64")
    else:
      if self.num_colors < self.n:
        raise ValueError("The number of colors must be at least the number of "
                         "objects or 1.")
      color_index = utils.random_int_k_from_n(
          key2,
          num_samples=n,
          n=self.num_colors,
          k=self.n
      )
    images = self._batch_render(particles, particles_radius, color_index)
    return images, dict(offset=offset, color_index=color_index)


class TwoBodySystem(NBodySystem):
  """N-body system with N = 2."""

  def __init__(
      self,
      m_range: utils.BoxRegion,
      g_range: utils.BoxRegion,
      radius_range: utils.BoxRegion,
      provided_canvas_bounds: utils.BoxRegion,
      **kwargs):
    self.radius_range = radius_range
    super().__init__(n=2, space_dims=2,
                     m_range=m_range,
                     g_range=g_range,
                     provided_canvas_bounds=provided_canvas_bounds,
                     **kwargs)

  def sample_y(
      self,
      num_samples: int,
      params: utils.Params,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> phase_space.PhaseSpace:
    pos = jax.random.uniform(rng_key, [num_samples, self.n])
    pos = self.radius_range.convert_from_unit_interval(pos)
    r = jnp.sqrt(jnp.sum(pos ** 2, axis=-1))

    vel = jnp.flip(pos, axis=-1) / (2 * r[..., None] ** 1.5)
    vel = vel * jnp.asarray([1.0, -1.0]).reshape([1, 2])

    pos = jnp.repeat(pos.reshape([num_samples, 1, -1]), repeats=self.n, axis=1)
    vel = jnp.repeat(vel.reshape([num_samples, 1, -1]), repeats=self.n, axis=1)

    pos = pos * jnp.asarray([1.0, -1.0]).reshape([1, 2, 1])
    vel = vel * jnp.asarray([1.0, -1.0]).reshape([1, 2, 1])
    pos = pos.reshape([num_samples, -1])
    vel = vel.reshape([num_samples, -1])
    return phase_space.PhaseSpace(position=pos, momentum=vel)


class ThreeBody2DSystem(NBodySystem):
  """N-body system with N = 3 in two dimensions."""

  def __init__(
      self,
      m_range: utils.BoxRegion,
      g_range: utils.BoxRegion,
      radius_range: utils.BoxRegion,
      provided_canvas_bounds: utils.BoxRegion,
      **kwargs):
    self.radius_range = radius_range
    super().__init__(n=3, space_dims=2,
                     m_range=m_range,
                     g_range=g_range,
                     provided_canvas_bounds=provided_canvas_bounds,
                     **kwargs)

  def sample_y(
      self,
      num_samples: int,
      params: utils.Params,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> phase_space.PhaseSpace:
    theta = 2 * jnp.pi / 3
    rot = jnp.asarray([[jnp.cos(theta), - jnp.sin(theta)],
                       [jnp.sin(theta), jnp.cos(theta)]])
    p1 = 2 * jax.random.uniform(rng_key, [num_samples, 2]) - 1.0
    r = jax.random.uniform(rng_key, [num_samples])
    r = self.radius_range.convert_from_unit_interval(r)

    p1 *= (r / jnp.linalg.norm(p1, axis=-1))[:, None]
    p2 = jnp.matmul(p1, rot.T)
    p3 = jnp.matmul(p2, rot.T)
    p = jnp.concatenate([p1, p2, p3], axis=-1)

    # scale factor to get circular trajectories
    factor = jnp.sqrt(jnp.sin(jnp.pi / 3)/(2 * jnp.cos(jnp.pi / 6) **2))
    # velocity that yields a circular orbit
    v1 = jnp.flip(p1, axis=-1) * factor / r[:, None]**1.5
    v2 = jnp.matmul(v1, rot.T)
    v3 = jnp.matmul(v2, rot.T)
    v = jnp.concatenate([v1, v2, v3], axis=-1)
    return phase_space.PhaseSpace(position=p, momentum=v)
