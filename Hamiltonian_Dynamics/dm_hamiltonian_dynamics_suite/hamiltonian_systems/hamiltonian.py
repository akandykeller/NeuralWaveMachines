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
"""A module with the abstract class for Hamiltonian systems."""
import abc
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from dm_hamiltonian_dynamics_suite.hamiltonian_systems import phase_space
from dm_hamiltonian_dynamics_suite.hamiltonian_systems import utils

import jax
import jax.numpy as jnp
import jax.random as jnr
from scipy import integrate

Integrator = Callable[
    [
        Union[phase_space.HamiltonianFunction,
              phase_space.SymplecticTangentFunction],  # dy_dt
        Union[jnp.ndarray, phase_space.PhaseSpace],  # y0
        Union[float, jnp.ndarray],  # t0
        Union[float, jnp.ndarray],  # dt
        int,  # num_steps
        int   # steps_per_dt
    ],
    Tuple[jnp.ndarray, phase_space.PhaseSpace]
]


class HamiltonianSystem(abc.ABC):
  """General class to represent Hamiltonian Systems and simulate them."""

  def __init__(
      self,
      system_dims: int,
      randomize_canvas_location: bool = True,
      random_canvas_extra_ratio: float = 0.4,
      try_analytic_solution: bool = True,
      friction: float = 0.0,
      method: Union[str, Integrator] = "scipy",
      num_colors: int = 6,
      image_resolution: int = 32,
      dtype: str = "float32",
      steps_per_dt: int = 1,
      stiff: bool = False,
      extra_ivp_kwargs: Optional[Mapping[str, Union[str, int, float]]] = None):
    """Initializes some global properties.

    Args:
      system_dims: Dimensionality of the positions (not joint state).
      randomize_canvas_location: Whether to add random offset to images.
      random_canvas_extra_ratio: How much to be the extra random ofsset.
      try_analytic_solution: If True, first tries to solve the system
        analytically possible. If False, alway integrates systems numerically.
      friction: This changes the dynamics to non-conservative. The new
          dynamics are formulated as follows:
            dq/dt = dH/dp
            dp/dt = - dH/dq - friction * dH/dp
          This implies that the Hamiltonian energy decreases over time:
            dH/dt = dH/dq^T dq/dt + dH/dp^T dp/dt = - friction * ||dH/dp||_2^2
      method: "scipy" or a callable of type `Integrator`.
      num_colors: The number of possible colors to use for rendering.
      image_resolution: For generated images their resolution.
      dtype: What dtype to use for the generated data.
      steps_per_dt: Number of inner steps to use per a single observation dt.
      stiff: Whether the problem represents a stiff system.
      extra_ivp_kwargs: Extra arguments to the scipy solver.
    Raises:
      ValueError: if `dtype` is not 'float32' or 'float64'.
    """
    self._system_dims = system_dims
    self._randomize_canvas_location = randomize_canvas_location
    self._random_canvas_extra_ratio = random_canvas_extra_ratio
    self._try_analytic_solution = try_analytic_solution
    self._friction = friction
    self._method = method
    self._num_colors = num_colors
    self._resolution = image_resolution
    self._dtype = dtype
    self._stiff = stiff
    self._steps_per_dt = steps_per_dt
    if dtype == "float64":
      self._scipy_ivp_kwargs = dict(rtol=1e-12, atol=1e-12)
    elif dtype == "float32":
      self._scipy_ivp_kwargs = dict(rtol=1e-9, atol=1e-9)
    else:
      raise ValueError("Currently we only support float64 and float32 dtypes.")
    if stiff:
      self._scipy_ivp_kwargs["method"] = "Radau"
    if extra_ivp_kwargs is not None:
      self._scipy_ivp_kwargs.update(extra_ivp_kwargs)

  @property
  def system_dims(self) -> int:
    return self._system_dims

  @property
  def randomize_canvas_location(self) -> bool:
    return self._randomize_canvas_location

  @property
  def random_canvas_extra_ratio(self) -> float:
    return self._random_canvas_extra_ratio

  @property
  def try_analytic_solution(self) -> bool:
    return self._try_analytic_solution

  @property
  def friction(self) -> float:
    return self._friction

  @property
  def method(self):
    return self._method

  @property
  def num_colors(self) -> int:
    return self._num_colors

  @property
  def resolution(self) -> int:
    return self._resolution

  @property
  def dtype(self) -> str:
    return self._dtype

  @property
  def stiff(self) -> bool:
    return self._stiff

  @property
  def steps_per_dt(self) -> int:
    return self._steps_per_dt

  @property
  def scipy_ivp_kwargs(self) -> Mapping[str, Union[str, int, float]]:
    return self._scipy_ivp_kwargs

  @abc.abstractmethod
  def parametrized_hamiltonian(
      self,
      t: jnp.ndarray,
      y: phase_space.PhaseSpace,
      params: utils.Params,
      **kwargs: Any
  ) -> jnp.ndarray:
    """Calculates the Hamiltonian."""

  def hamiltonian_from_params(
      self,
      params: utils.Params,
      **kwargs: Any
  ) -> phase_space.HamiltonianFunction:
    def hamiltonian(t: jnp.ndarray, y: phase_space.PhaseSpace) -> jnp.ndarray:
      return self.parametrized_hamiltonian(t, y, params, **kwargs)
    return hamiltonian

  @abc.abstractmethod
  def sample_y(
      self,
      num_samples: int,
      params: utils.Params,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> phase_space.PhaseSpace:
    """Samples randomly initial states."""

  @abc.abstractmethod
  def sample_params(
      self,
      num_samples: int,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> utils.Params:
    """Samples randomly parameters."""

  @abc.abstractmethod
  def simulate_analytically(
      self,
      y0: phase_space.PhaseSpace,
      t0: utils.FloatArray,
      t_eval: jnp.ndarray,
      params: utils.Params,
      **kwargs: Any
  ) -> Optional[phase_space.PhaseSpace]:
    """If analytic solution exist returns it, else returns None."""

  def simulate_analytically_dt(
      self,
      y0: phase_space.PhaseSpace,
      t0: utils.FloatArray,
      dt: utils.FloatArray,
      num_steps: int,
      params: utils.Params,
      **kwargs: Any
  ) -> Optional[phase_space.PhaseSpace]:
    """Same as `simulate_analytically` but uses `dt` and `num_steps`."""
    t_eval = utils.dt_to_t_eval(t0, dt, num_steps)
    return self.simulate_analytically(y0, t0, t_eval, params, **kwargs)

  @abc.abstractmethod
  def canvas_bounds(self) -> utils.BoxRegion:
    """Returns the limits of the canvas for rendering."""

  def random_offset_bounds(self) -> utils.BoxRegion:
    """Returns any extra randomized offset that can be given to the canvas."""
    extra_size = self.random_canvas_extra_ratio * self.canvas_bounds().size / 2
    return utils.BoxRegion(
        minimum=-extra_size,
        maximum=extra_size
    )

  def full_canvas_bounds(self) -> utils.BoxRegion:
    if self.randomize_canvas_location:
      return utils.BoxRegion(
          self.canvas_bounds().min + self.random_offset_bounds().min,
          self.canvas_bounds().max + self.random_offset_bounds().max,
      )
    else:
      return self.canvas_bounds()

  @abc.abstractmethod
  def canvas_position(
      self,
      position: jnp.ndarray,
      params: utils.Params
  ) -> jnp.ndarray:
    """Returns the canvas position given the position vectors and the parameters."""

  @abc.abstractmethod
  def render_trajectories(
      self,
      position: jnp.ndarray,
      params: utils.Params,
      rng_key: jnp.ndarray,
      **kwargs: Any
  ) -> Tuple[jnp.ndarray, utils.Params]:
    """Renders the positions q into an image."""

  def simulate_scipy(
      self,
      y0: phase_space.PhaseSpace,
      t0: utils.FloatArray,
      t_eval: jnp.ndarray,
      params: utils.Params,
      ivp_kwargs=None,
      **kwargs: Any
  ) ->  phase_space.PhaseSpace:
    """Simulates the system using scipy.integrate.solve_ivp."""
    t_span = (t0, float(t_eval[-1]))
    y0 = jnp.concatenate([y0.q, y0.p], axis=-1)
    y_shape = y0.shape
    y0 = y0.reshape([-1])
    hamiltonian = self.hamiltonian_from_params(params, **kwargs)
    @jax.jit
    def fun(t, y):
      f = phase_space.poisson_bracket_with_q_and_p(hamiltonian)
      dy = f(t, phase_space.PhaseSpace.from_state(y.reshape(y_shape)))
      if self.friction != 0.0:
        friction_term = phase_space.TangentPhaseSpace(
            position=jnp.zeros_like(dy.position),
            momentum=-self.friction * dy.position)
        dy = dy + friction_term
      return dy.single_state.reshape([-1])
    kwargs = dict(**self.scipy_ivp_kwargs)
    kwargs.update(ivp_kwargs or dict())
    solution = integrate.solve_ivp(
        fun=fun,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        **kwargs)
    y_final = solution.y.reshape(y_shape + (t_eval.size,))
    return phase_space.PhaseSpace.from_state(jnp.moveaxis(y_final, -1, 0))

  def simulate_scipy_dt(
      self,
      y0: phase_space.PhaseSpace,
      t0: utils.FloatArray,
      dt: utils.FloatArray,
      num_steps: int,
      params: utils.Params,
      ivp_kwargs=None,
      **kwargs: Any
  ) ->  phase_space.PhaseSpace:
    """Same as `simulate_scipy` but uses `dt` and `num_steps`."""
    t_eval = utils.dt_to_t_eval(t0, dt, num_steps)
    return self.simulate_scipy(y0, t0, t_eval, params, ivp_kwargs, **kwargs)

  def simulate_integrator(
      self,
      y0: phase_space.PhaseSpace,
      t0: utils.FloatArray,
      t_eval: jnp.ndarray,
      params: utils.Params,
      method: Union[str, Integrator],
      **kwargs: Any
  ) -> phase_space.PhaseSpace:
    """Simulates the system using an integrator from integrators.py module."""
    return self.simulate_integrator_dt(
        y0=y0,
        t0=t0,
        dt=utils.t_eval_to_dt(t0, t_eval),
        params=params,
        method=method,
        **kwargs
    )

  def simulate_integrator_dt(
      self,
      y0: phase_space.PhaseSpace,
      t0: Union[float, jnp.ndarray],
      dt: Union[float, jnp.ndarray],
      params: utils.Params,
      method: Integrator,
      num_steps: Optional[int] = None,
      **kwargs: Any
  ) -> phase_space.PhaseSpace:
    """Same as `simulate_integrator` but uses `dt` and `num_steps`."""
    hamiltonian = self.hamiltonian_from_params(params, **kwargs)
    if self.friction == 0.0:
      return method(
          hamiltonian,
          y0,
          t0,
          dt,
          num_steps,
          self.steps_per_dt,
      )[1]
    else:
      def dy_dt(t: jnp.ndarray, y: phase_space.PhaseSpace):
        f = phase_space.poisson_bracket_with_q_and_p(hamiltonian)
        dy = f(t, y)
        friction_term = phase_space.TangentPhaseSpace(
            position=jnp.zeros_like(dy.position),
            momentum=-self.friction * dy.position)
        return dy + friction_term

      return method(
          dy_dt,
          y0,
          t0,
          dt,
          num_steps,
          self.steps_per_dt,
      )[1]

  def generate_trajectories(
      self,
      y0: phase_space.PhaseSpace,
      t0: utils.FloatArray,
      t_eval: jnp.ndarray,
      params: utils.Params,
      **kwargs: Any
  ) -> phase_space.PhaseSpace:
    """Generates trajectories of the system in phase space.

    Args:
      y0: Initial state.
      t0: The time instance of the initial state y0.
      t_eval: Times at which to return the computed solution.
      params: Any parameters of the Hamiltonian.
      **kwargs: Any extra things that go into the hamiltonian.

    Returns:
      A phase_space.PhaseSpace instance of size NxTxD.
    """
    return self.generate_trajectories_dt(
        y0=y0,
        t0=t0,
        dt=utils.t_eval_to_dt(t0, t_eval),
        params=params,
        **kwargs
    )

  def generate_trajectories_dt(
      self,
      y0: phase_space.PhaseSpace,
      t0: utils.FloatArray,
      dt: utils.FloatArray,
      params: utils.Params,
      num_steps_forward: int,
      include_t0: bool = False,
      num_steps_backward: int = 0,
      **kwargs: Any
  ) -> phase_space.PhaseSpace:
    """Same as `generate_trajectories` but uses `dt` and `num_steps`."""
    if num_steps_forward < 0 or num_steps_backward < 0:
      raise ValueError("num_steps_forward and num_steps_backward can not be "
                       "negative.")
    if num_steps_forward == 0 and num_steps_backward == 0:
      raise ValueError("You need one of num_steps_forward or "
                       "num_of_steps_backward to be positive.")
    if num_steps_forward > 0 and num_steps_backward > 0 and not include_t0:
      raise ValueError("When both num_steps_forward and num_steps_backward are "
                       "positive include_t0 should be True.")

    if self.try_analytic_solution and num_steps_backward == 0:
      # Try to use analytical solution
      y = self.simulate_analytically_dt(y0, t0, dt, num_steps_forward, params,
                                        **kwargs)
      if y is not None:
        return y
    if self.method == "scipy":
      if num_steps_backward > 0:
        raise NotImplementedError()
      return self.simulate_scipy_dt(y0, t0, dt, num_steps_forward, params,
                                    **kwargs)
    yts = []
    if num_steps_backward > 0:
      yt = self.simulate_integrator_dt(
          y0=y0,
          t0=t0,
          dt=-dt,
          params=params,
          method=self.method,
          num_steps=num_steps_backward,
          **kwargs)
      yt = jax.tree_map(lambda x: jnp.flip(x, axis=0), yt)
      yts.append(yt)
    if include_t0:
      yts.append(jax.tree_map(lambda x: x[None], y0))
    if num_steps_forward > 0:
      yt = self.simulate_integrator_dt(
          y0=y0,
          t0=t0,
          dt=dt,
          params=params,
          method=self.method,
          num_steps=num_steps_forward,
          **kwargs)
      yts.append(yt)
    if len(yts) > 1:
      return jax.tree_multimap(lambda *a: jnp.concatenate(a, axis=0), *yts)
    else:
      return yts[0]

  def generate_and_render(
      self,
      num_trajectories: int,
      rng_key: jnp.ndarray,
      t0: utils.FloatArray,
      t_eval: utils.FloatArray,
      y0: Optional[phase_space.PhaseSpace] = None,
      params: Optional[utils.Params] = None,
      within_canvas_bounds: bool = True,
      **kwargs: Any
  ) -> Mapping[str, Any]:
    """Generates trajectories and renders them.

    Args:
      num_trajectories: The number of trajectories to generate.
      rng_key: PRNG key for sampling any random numbers.
      t0: The time instance of the initial state y0.
      t_eval: Times at which to return the computed solution.
      y0: Initial state. If None will be sampled with `self.sample_y`
      params: Parameters of the Hamiltonian. If None will be sampled with
        `self.sample_params`
      within_canvas_bounds: Re-samples y0 until the trajectories is within
        the canvas bounds.
      **kwargs: Any extra things that go into the hamiltonian.

    Returns:
      A dictionary containing the following elements:
        "x": A numpy array representation of the PhaseSpace vector.
        "dx_dt": The time derivative of "x".
        "image": An image representation of the state.
        "other": A dict of other parameters of the system that are not part of
        the state.
    """
    return self.generate_and_render_dt(
        num_trajectories=num_trajectories,
        rng_key=rng_key,
        t0=t0,
        dt=utils.t_eval_to_dt(t0, t_eval),
        y0=y0,
        params=params,
        within_canvas_bounds=within_canvas_bounds,
        **kwargs
    )

  def generate_and_render_dt(
      self,
      num_trajectories: int,
      rng_key: jnp.ndarray,
      t0: utils.FloatArray,
      dt: utils.FloatArray,
      num_steps: Optional[int] = None,
      y0: Optional[phase_space.PhaseSpace] = None,
      params: Optional[utils.Params] = None,
      within_canvas_bounds: bool = True,
      **kwargs: Any
  ) -> Mapping[str, Any]:
    """Same as `generate_and_render` but uses `dt` and `num_steps`."""
    if within_canvas_bounds and (y0 is not None or params is not None):
      raise ValueError("Within canvas bounds is valid only when y0 and params "
                       "are None.")
    if params is None:
      rng_key, key = jnr.split(rng_key)
      params = self.sample_params(num_trajectories, rng_key, **kwargs)
    if y0 is None:
      rng_key, key = jnr.split(rng_key)
      y0 = self.sample_y(num_trajectories, params, key, **kwargs)

    # Generate the phase-space trajectories
    x = self.generate_trajectories_dt(y0, t0, dt, params, num_steps, **kwargs)
    # Make batch leading dimension
    x = jax.tree_map(lambda x_: jnp.swapaxes(x_, 0, 1), x)
    x = jax.tree_multimap(lambda i, j: jnp.concatenate([i[:, None], j], axis=1),
                          y0, x)
    if within_canvas_bounds:
      # Check for valid trajectories
      valid = []
      while len(valid) < num_trajectories:
        for idx in range(x.q.shape[0]):
          x_idx, params_idx = jax.tree_map(lambda a, i=idx: a[i], (x, params))
          position = self.canvas_position(x_idx.q, params_idx)
          if (jnp.all(position >= self.canvas_bounds().min) and
              jnp.all(position <= self.canvas_bounds().max)):
            valid.append((x_idx, params_idx))
        if len(valid) == num_trajectories:
          break
        new_trajectories = num_trajectories - len(valid)
        print(f"Generating {new_trajectories} new trajectories.")
        rng_key, key = jnr.split(rng_key)
        params = self.sample_params(new_trajectories, rng_key, **kwargs)
        rng_key, key = jnr.split(rng_key)
        y0 = self.sample_y(new_trajectories, params, key, **kwargs)
        x = self.generate_trajectories_dt(y0, t0, dt, params, num_steps,
                                          **kwargs)
        x = jax.tree_map(lambda x_: jnp.swapaxes(x_, 0, 1), x)
        x = jax.tree_multimap(lambda i, j:  # pylint:disable=g-long-lambda
                              jnp.concatenate([i[:, None], j], axis=1), y0, x)
      x, params = jax.tree_multimap(lambda *args: jnp.stack(args, axis=0),
                                    *valid)

    hamiltonian = self.hamiltonian_from_params(params, **kwargs)
    df_dt = jax.vmap(phase_space.poisson_bracket_with_q_and_p(hamiltonian),
                     in_axes=[0, 1], out_axes=1)
    if isinstance(dt, float):
      dt = jnp.asarray([dt] * num_steps, dtype=x.q.dtype)
    t0 = jnp.asarray(t0).astype(dt.dtype)
    t = jnp.cumsum(jnp.concatenate([t0[None], dt], axis=0), axis=0)
    dx_dt = df_dt(t, x)
    rng_key, key = jnr.split(rng_key)
    image, extra = self.render_trajectories(x.q, params, rng_key, **kwargs)
    params.update(extra)
    return dict(x=x.single_state, dx_dt=dx_dt.single_state,
                image=image, other=params)


class TimeIndependentHamiltonianSystem(HamiltonianSystem):
  """A Hamiltonian system where the energy does not depend on time."""

  @abc.abstractmethod
  def _hamiltonian(
      self,
      y: phase_space.PhaseSpace,
      params: utils.Params,
      **kwargs: Any
  ) -> jnp.ndarray:
    """Computes the time independent Hamiltonian."""

  def parametrized_hamiltonian(
      self,
      t: jnp.ndarray,
      y: phase_space.PhaseSpace,
      params: utils.Params,
      **kwargs: Any
  ) -> jnp.ndarray:
    return self._hamiltonian(y, params, **kwargs)
