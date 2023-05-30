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
"""A module with all Hamiltonian systems that have analytic solutions."""
from typing import Any, Optional, Tuple

from dm_hamiltonian_dynamics_suite.hamiltonian_systems import hamiltonian
from dm_hamiltonian_dynamics_suite.hamiltonian_systems import phase_space
from dm_hamiltonian_dynamics_suite.hamiltonian_systems import utils

import jax.numpy as jnp
import jax.random as jnr


class PotentialFreeSystem(hamiltonian.TimeIndependentHamiltonianSystem):
  """A system where the potential energy is 0 and the kinetic is quadratic.

  Parameters:
    matrix - a positive semi-definite matrix used for the kinetic quadratic.

  The Hamiltonian is:
      p^T M p / 2

  Initial state parameters:
    min_radius - the minimum radius to sample from
    max_radius - the maximum radius to sample from
  """

  def __init__(
      self,
      system_dims: int,
      eigen_values_range: utils.BoxRegion,
      init_vector_range: utils.BoxRegion,
      **kwargs):
    super().__init__(system_dims=system_dims, **kwargs)
    if eigen_values_range.dims != 0 and eigen_values_range.dims != system_dims:
      raise ValueError(f"The eigen_values_range must be of the same dimensions "
                       f"as the system dimensions, but is "
                       f"{eigen_values_range.dims}.")
    if init_vector_range.dims != 0 and init_vector_range.dims != system_dims:
      raise ValueError(f"The init_vector_range must be of the same dimensions "
                       f"as the system dimensions, but is "
                       f"{init_vector_range.dims}.")
    self.eigen_values_range = eigen_values_range
    self.init_vector_range = init_vector_range

  def _hamiltonian(
      self,
      y: phase_space.PhaseSpace,
      params: utils.Params,
      **kwargs: Any
  ) -> jnp.ndarray:
    assert len(params) == 1
    matrix = params["matrix"]
    potential = 0
    kinetic = jnp.sum(jnp.matmul(y.p, matrix) * y.p, axis=-1) / 2
    return potential + kinetic

  def sample_y(
      self,
      num_samples: int,
      params: utils.Params,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> phase_space.PhaseSpace:
    # Sample random state
    y = jnr.uniform(rng_key, [num_samples, 2 * self.system_dims],
                    dtype=self.dtype)
    y = self.init_vector_range.convert_from_unit_interval(y)
    return phase_space.PhaseSpace.from_state(y)

  def sample_params(
      self,
      num_samples: int,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> utils.Params:
    key1, key2 = jnr.split(rng_key)
    matrix_shape = [num_samples, self.system_dims, self.system_dims]
    gaussian = jnr.normal(key1, matrix_shape)
    q, _ = jnp.linalg.qr(gaussian)
    eigs = jnr.uniform(key2, [num_samples, self.system_dims])
    eigs = self.eigen_values_range.convert_from_unit_interval(eigs)
    q_eigs = q * eigs[..., None]
    matrix = jnp.matmul(q_eigs, jnp.swapaxes(q_eigs, -2, -1))
    return dict(matrix=matrix)

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
    assert len(params) == 1
    matrix = params["matrix"]
    t = utils.expand_to_rank_right(t_eval - t0, y0.q.ndim + 1)
    q = y0.q[None] + utils.vecmul(matrix, y0.p)[None] * t
    p = y0.p[None] * jnp.ones_like(t)
    return phase_space.PhaseSpace(position=q, momentum=p)

  def canvas_bounds(self) -> utils.BoxRegion:
    raise NotImplementedError()

  def canvas_position(
      self,
      position: jnp.ndarray,
      params: utils.Params
  ) -> jnp.ndarray:
    raise NotImplementedError()

  def render_trajectories(
      self,
      position: jnp.ndarray,
      params: utils.Params,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> Tuple[jnp.ndarray, utils.Params]:
    raise NotImplementedError()


class KineticFreeSystem(PotentialFreeSystem):
  """A system where the kinetic energy is 0 and the potential is quadratic.

  Parameters:
    matrix - a positive semi-definite matrix used for the potential quadratic.

  The Hamiltonian is:
      q^T M q / 2

  Initial state parameters:
    min_radius - the minimum radius to sample from
    max_radius - the maximum radius to sample from
  """

  def _hamiltonian(
      self,
      y: phase_space.PhaseSpace,
      params: utils.Params,
      **kwargs: Any
  ) -> jnp.ndarray:
    assert len(params) == 1
    matrix = params["matrix"]
    potential = jnp.sum(jnp.matmul(y.q, matrix) * y.q, axis=-1) / 2
    kinetic = 0
    return potential + kinetic

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
    assert len(params) == 1
    matrix = params["matrix"]
    t = utils.expand_to_rank_right(t_eval - t0, y0.q.ndim + 1)
    q = y0.q[None] * jnp.ones_like(t)
    p = y0.p[None] - utils.vecmul(matrix, y0.q)[None] * t
    return phase_space.PhaseSpace(position=q, momentum=p)

  def canvas_bounds(self) -> utils.BoxRegion:
    raise NotImplementedError()

  def canvas_position(
      self,
      position: jnp.ndarray,
      params: utils.Params
  ) -> jnp.ndarray:
    raise NotImplementedError()

  def render_trajectories(
      self,
      position: jnp.ndarray,
      params: utils.Params,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> Tuple[jnp.ndarray, utils.Params]:
    raise NotImplementedError()
