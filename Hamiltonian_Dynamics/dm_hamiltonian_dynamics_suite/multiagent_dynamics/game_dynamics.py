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
"""Continuous-time two player zero-sum game dynamics."""
from typing import Any, Mapping

from dm_hamiltonian_dynamics_suite.hamiltonian_systems import utils
import jax.numpy as jnp
import jax.random as jnr
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy
from open_spiel.python.egt import dynamics as egt_dynamics
from open_spiel.python.egt import utils as egt_utils
from open_spiel.python.pybind11 import pyspiel


def sample_from_simplex(
    num_samples: int,
    rng: jnp.ndarray,
    dim: int = 3,
    vmin: float = 0.
) -> jnp.ndarray:
  """Samples random points from a k-simplex. See D. B. Rubin (1981), p131."""
  # This is a jax version of open_spiel.python.egt.utils.sample_from_simplex.
  assert vmin >= 0.
  p = jnr.uniform(rng, shape=(num_samples, dim - 1))
  p = jnp.sort(p, axis=1)
  p = jnp.hstack((jnp.zeros((num_samples, 1)), p, jnp.ones((num_samples, 1))))
  return (p[:, 1:] - p[:, 0:-1]) * (1 - 2 * vmin) + vmin


def get_payoff_tensor(game_name: str) -> jnp.ndarray:
  """Returns the payoff tensor of a game."""
  game = pyspiel.extensive_to_tensor_game(pyspiel.load_game(game_name))
  assert game.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM
  payoff_tensor = egt_utils.game_payoffs_array(game)
  return payoff_tensor


def tile_array(a: jnp.ndarray, b0: int, b1: int) -> jnp.ndarray:
  r, c = a.shape  # number of rows/columns
  rs, cs = a.strides  # row/column strides
  x = as_strided(a, (r, b0, c, b1), (rs, 0, cs, 0))  # view a as larger 4D array
  return x.reshape(r * b0, c * b1)  # create new 2D array


class ZeroSumGame:
  """Generate trajectories from zero-sum game dynamics."""

  def __init__(
      self,
      game_name: str,
      dynamics: str = 'replicator',
      method: str = 'scipy'):
    self.payoff_tensor = get_payoff_tensor(game_name)
    assert self.payoff_tensor.shape[0] == 2, 'Only supports two-player games.'
    dyn_fun = getattr(egt_dynamics, dynamics)
    self.dynamics = egt_dynamics.MultiPopulationDynamics(
        self.payoff_tensor, dyn_fun)
    self.method = method
    self.scipy_ivp_kwargs = dict(rtol=1e-12, atol=1e-12)

  def sample_x0(self, num_samples: int, rng_key: jnp.ndarray) -> jnp.ndarray:
    """Samples initial states."""
    nrows, ncols = self.payoff_tensor.shape[1:]
    key1, key2 = jnr.split(rng_key)
    x0_1 = sample_from_simplex(num_samples, key1, dim=nrows)
    x0_2 = sample_from_simplex(num_samples, key2, dim=ncols)
    x0 = jnp.hstack((x0_1, x0_2))
    return x0

  def generate_trajectories(
      self,
      x0: jnp.ndarray,
      t0: utils.FloatArray,
      t_eval: jnp.ndarray
  ) -> jnp.ndarray:
    """Generates trajectories of the system in phase space.

    Args:
      x0: Initial state.
      t0: The time instance of the initial state y0.
      t_eval: Times at which to return the computed solution.

    Returns:
      Trajectories of size BxTxD (batch, time, phase-space-dim).
    """
    if self.method == 'scipy':
      x0_shape = x0.shape

      def fun(_, y):
        y = y.reshape(x0_shape)
        y_next = np.apply_along_axis(self.dynamics, -1, y)
        return y_next.reshape([-1])

      t_span = (t0, float(t_eval[-1]))
      solution = scipy.integrate.solve_ivp(
          fun=fun,
          t_span=t_span,
          y0=x0.reshape([-1]),  # Scipy requires flat input.
          t_eval=t_eval,
          **self.scipy_ivp_kwargs)
      x = solution.y.reshape(x0_shape + (t_eval.size,))
      x = np.moveaxis(x, -1, 1)  # Make time 2nd dimension.
    else:
      raise ValueError(f'Method={self.method} not supported.')
    return x

  def render_trajectories(self, x: jnp.ndarray) -> jnp.ndarray:
    """Maps from policies to joint-policy space."""
    nrows, ncols = self.payoff_tensor.shape[1:]
    x_1 = x[..., :nrows]
    x_2 = x[..., nrows:]
    x_1 = x_1.repeat(ncols, axis=-1).reshape(x.shape[:-1] + (nrows, ncols,))
    x_2 = x_2.repeat(nrows, axis=-1).reshape(x.shape[:-1] + (nrows, ncols,))
    x_2 = x_2.swapaxes(-2, -1)
    image = x_1 * x_2

    # Rescale to 32 x 32 from the original 2x2 or 3x3 data by expanding the
    # matrix to the nearest to 32 multiple of 2 or 3, evenly tiling it with the
    # original values, and then taking a 32x32 top left slice of it
    temp_image = [
        tile_array(x, np.ceil(32 / x.shape[0]).astype('int'),
                   np.ceil(32 / x.shape[1]).astype('int'))[:32, :32]
        for x in np.squeeze(image)
    ]
    image = np.stack(temp_image)
    image = np.repeat(np.expand_dims(image, -1), 3, axis=-1)

    return image[None, ...]

  def generate_and_render(
      self,
      num_trajectories: int,
      rng_key: jnp.ndarray,
      t0: utils.FloatArray,
      t_eval: utils.FloatArray
  ) -> Mapping[str, Any]:
    """Generates trajectories and renders them.

    Args:
      num_trajectories: The number of trajectories to generate.
      rng_key: PRNG key for sampling any random numbers.
      t0: The time instance of the initial state y0.
      t_eval: Times at which to return the computed solution.

    Returns:
      A dictionary containing the following elements:
        'x': A numpy array representation of the phase space vector.
        'dx_dt': The time derivative of 'x'.
        'image': An image representation of the state.
    """
    rng_key, key = jnr.split(rng_key)
    x0 = self.sample_x0(num_trajectories, key)
    x = self.generate_trajectories(x0, t0, t_eval)
    x = np.concatenate([x0[:, None], x], axis=1)  # Add initial state.
    dx_dt = np.apply_along_axis(self.dynamics, -1, x)
    image = self.render_trajectories(x)

    return dict(x=x, dx_dt=dx_dt, image=image)

  def generate_and_render_dt(
      self,
      num_trajectories: int,
      rng_key: jnp.ndarray,
      t0: utils.FloatArray,
      dt: utils.FloatArray,
      num_steps: int
  ) -> Mapping[str, Any]:
    """Same as `generate_and_render` but uses `dt` and `num_steps`."""
    t_eval = utils.dt_to_t_eval(t0, dt, num_steps)
    return self.generate_and_render(num_trajectories, rng_key, t0, t_eval)
