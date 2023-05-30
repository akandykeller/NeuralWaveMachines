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
"""Module for the PhaseSpace class."""
import functools
from typing import Callable, Type, Union

import jax
from jax import numpy as jnp
from jax import tree_util


class PhaseSpace(object):
  """Holds a pair of position and momentum for a Hamiltonian System."""

  def __init__(self, position: jnp.ndarray, momentum: jnp.ndarray):
    self._position = position
    self._momentum = momentum

  @property
  def position(self) -> jnp.ndarray:
    """The position element of the phase space."""
    return self._position

  @property
  def momentum(self) -> jnp.ndarray:
    """The momentum element of the phase space."""
    return self._momentum

  @property
  def q(self) -> jnp.ndarray:
    """A shorthand for the position element of the phase space."""
    return self._position

  @property
  def p(self) -> jnp.ndarray:
    """A shorthand for the momentum element of the phase space."""
    return self._momentum

  @property
  def single_state(self) -> jnp.ndarray:
    """Returns the concatenation of position and momentum."""
    return jnp.concatenate([self.q, self.p], axis=-1)

  @property
  def ndim(self) -> int:
    """Returns the number of dimensions of the position array."""
    return self.q.ndim

  @classmethod
  def from_state(cls: Type["PhaseSpace"], state: jnp.ndarray) -> "PhaseSpace":
    q, p = jnp.split(state, 2, axis=-1)
    return cls(position=q, momentum=p)

  def __str__(self) -> str:
    return f"{type(self).__name__}(q={self.position}, p={self.momentum})"

  def __repr__(self) -> str:
    return self.__str__()


class TangentPhaseSpace(PhaseSpace):
  """Represents the tangent space to PhaseSpace."""

  def __add__(
      self,
      other: Union[PhaseSpace, "TangentPhaseSpace"],
  ) -> Union[PhaseSpace, "TangentPhaseSpace"]:
    if isinstance(other, TangentPhaseSpace):
      return TangentPhaseSpace(position=self.q + other.q,
                               momentum=self.p + other.p)
    elif isinstance(other, PhaseSpace):
      return PhaseSpace(position=self.q + other.q,
                        momentum=self.p + other.p)
    else:
      raise ValueError(f"Can not add TangentPhaseSpace and {type(other)}.")

  def __radd__(
      self,
      other: Union[PhaseSpace, "TangentPhaseSpace"]
  ) -> Union[PhaseSpace, "TangentPhaseSpace"]:
    return self.__add__(other)

  def __mul__(self, other: jnp.ndarray) -> "TangentPhaseSpace":
    return TangentPhaseSpace(position=self.q * other,
                             momentum=self.p * other)

  def __rmul__(self, other):
    return self.__mul__(other)

  @classmethod
  def zero(cls: Type["TangentPhaseSpace"]) -> "TangentPhaseSpace":
    return cls(position=jnp.asarray(0.0), momentum=jnp.asarray(0.0))


HamiltonianFunction = Callable[
    [
        jnp.ndarray,  # t
        PhaseSpace,  # y
    ],
    jnp.ndarray  # H(t, y)
]

SymplecticTangentFunction = Callable[
    [
        jnp.ndarray,  # t
        PhaseSpace  # (q, p)
    ],
    TangentPhaseSpace  # (dH_dp, - dH_dq)
]

SymplecticTangentFunctionArray = Callable[
    [
        jnp.ndarray,  # t
        jnp.ndarray  # (q, p)
    ],
    jnp.ndarray  # (dH_dp, - dH_dq)
]


def poisson_bracket_with_q_and_p(
    f: HamiltonianFunction
) -> SymplecticTangentFunction:
  """Returns a function that computes the Poisson brackets {q,f} and {p,f}."""
  def bracket(t: jnp.ndarray, y: PhaseSpace) -> TangentPhaseSpace:
    # Use the summation trick for getting gradient
    # Note that the first argument to the hamiltonian is t
    grad = jax.grad(lambda *args: jnp.sum(f(*args)), argnums=1)(t, y)
    return TangentPhaseSpace(position=grad.p, momentum=-grad.q)
  return bracket


def transform_symplectic_tangent_function_using_array(
    func: SymplecticTangentFunction
) -> SymplecticTangentFunctionArray:
  @functools.wraps(func)
  def wrapped(t: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
    return func(t, PhaseSpace.from_state(state)).single_state
  return wrapped


tree_util.register_pytree_node(
    nodetype=PhaseSpace,
    flatten_func=lambda y: ((y.q, y.p), None),
    unflatten_func=lambda _, q_and_p: PhaseSpace(*q_and_p)
)

tree_util.register_pytree_node(
    nodetype=TangentPhaseSpace,
    flatten_func=lambda y: ((y.q, y.p), None),
    unflatten_func=lambda _, q_and_p: TangentPhaseSpace(*q_and_p)
)
