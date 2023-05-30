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
"""Module containing dataset configurations used for generation and some utility functions."""
import functools
import os
import shutil
from typing import Callable, Mapping, Any, TextIO, Generator, Tuple, Optional

from absl import logging

from dm_hamiltonian_dynamics_suite.hamiltonian_systems import ideal_double_pendulum
from dm_hamiltonian_dynamics_suite.hamiltonian_systems import ideal_mass_spring
from dm_hamiltonian_dynamics_suite.hamiltonian_systems import ideal_pendulum
from dm_hamiltonian_dynamics_suite.hamiltonian_systems import n_body
from dm_hamiltonian_dynamics_suite.hamiltonian_systems import utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

PipelineOutput = Optional[Tuple[
    Tuple[Mapping[str, jnp.ndarray], ...],
    Tuple[Mapping[str, jnp.ndarray], ...],
]]

try:
  from dm_hamiltonian_dynamics_suite.multiagent_dynamics import game_dynamics  # pylint: disable=g-import-not-at-top
  _OPEN_SPIEL_INSTALLED = True
except ModuleNotFoundError:
  _OPEN_SPIEL_INSTALLED = False


def open_spiel_available() -> bool:
  return _OPEN_SPIEL_INSTALLED


def set_up_folder(folder: str, overwrite: bool) -> None:
  """Sets up the folder needed for the dataset (optionally clearing it)."""
  if os.path.exists(folder):
    if overwrite:
      shutil.rmtree(folder)
      os.makedirs(folder)
  else:
    os.makedirs(folder)


def save_features(
    file: TextIO,
    example_dict: Mapping[str, Any],
    prefix: str = ""
) -> None:
  """Saves the features file used for loading."""
  for k, v in example_dict.items():
    if isinstance(v, dict):
      save_features(file, v, prefix=f"{prefix}{k}/")
    else:
      if isinstance(v, tf.Tensor):
        v = v.numpy()
      if isinstance(v, (np.ndarray, jnp.ndarray)):
        # int32 are promoted to int64
        if v.dtype == np.int32:
          file.write(f"{prefix}{k}, {v.shape}, {np.int64}\n")
        else:
          file.write(f"{prefix}{k}, {v.shape}, {v.dtype}\n")
      else:
        raise NotImplementedError(f"Currently the only supported feature types "
                                  f"are tf.Tensor, np.ndarray and jnp.ndarray. "
                                  f"Encountered value of type {type(v)}.")


def encode_example(example_dict: Mapping[str, Any]) -> Mapping[str, Any]:
  """Encodes a single trajectory into a TFRecord example."""
  result_dict = dict()
  for k, v in example_dict.items():
    if isinstance(v, tf.Tensor):
      v = v.numpy()
    if isinstance(v, dict):
      for ki, vi in encode_example(v).items():
        result_dict[f"{k}/{ki}"] = vi
    elif isinstance(v, (np.ndarray, jnp.ndarray)):
      if v.dtype == np.uint8:
        # We encode images to png
        if v.ndim == 4:
          # Since encode_png accepts only a single image for a batch of images
          # we just stack them over their first axis.
          v = v.reshape((-1,) + v.shape[-2:])
        image_string = tf.image.encode_png(v).numpy()
        result_dict[k] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_string]))
      elif v.dtype == np.int32:
        # int32 are promoted to int64
        value = v.reshape([-1]).astype(np.int64)
        result_dict[k] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=value))
      else:
        # Since tf.Records do not support reading float64, here for any values
        # we interpret them as int64 and store them in this format, in order
        # when reading to be able to recover the float64 values.
        value = v.reshape([-1]).view(np.int64)
        result_dict[k] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=value))
    else:
      raise NotImplementedError(f"Currently the only supported feature types "
                                f"are tf.Tensor, np.ndarray and jnp.ndarray. "
                                f"Encountered value of type {type(v)}.")
  return result_dict


def transform_dataset(
    generator: Generator[Mapping[str, Any], None, None],
    destination_folder: str,
    prefix: str,
    overwrite: bool
) -> None:
  """Copies the dataset from the source folder to the destination as a TFRecord dataset."""
  set_up_folder(destination_folder, overwrite)
  features_path = os.path.join(destination_folder, "features.txt")
  features_saved = False

  file_path = os.path.join(destination_folder, f"{prefix}.tfrecord")
  if os.path.exists(file_path):
    if not overwrite:
      logging.info("The file with prefix %s already exist. Skipping.", prefix)
      # We assume that the features file must be present in this case.
      return
    else:
      logging.info("The file with prefix %s already exist and overwrite=True."
                   " Deleting.", prefix)
      os.remove(file_path)
  with tf.io.TFRecordWriter(file_path) as writer:
    for element in generator:
      if not features_saved:
        with open(features_path, "w") as f:
          save_features(f, element)
        features_saved = True
      example = tf.train.Example(features=tf.train.Features(
          feature=encode_example(element)))
      writer.write(example.SerializeToString())


def generate_sample(
    index: int,
    system: n_body.hamiltonian.HamiltonianSystem,
    dt: float,
    num_steps: int,
    steps_per_dt: int
) -> Mapping[str, jnp.ndarray]:
  """Simulates a single trajectory of the system."""
  seed = np.random.randint(0, 2 * 32 -1)
  prng_key = jax.random.fold_in(jax.random.PRNGKey(seed), index)
  total_steps = num_steps * steps_per_dt
  total_dt = dt / steps_per_dt
  result = system.generate_and_render_dt(
      num_trajectories=1,
      rng_key=prng_key,
      t0=0.0,
      dt=total_dt,
      num_steps=total_steps)
  sub_sample_index = np.linspace(0.0, total_steps, num_steps + 1)
  sub_sample_index = sub_sample_index.astype("int64")
  def sub_sample(x):
    if x.ndim > 1 and x.shape[1] == total_steps + 1:
      return x[0, sub_sample_index]
    else:
      return x
  result = jax.tree_map(sub_sample, result)
  for k in result.keys():
    if "image" in k:
      result[k] = (result[k] * 255.0).astype("uint8")
  return result


def create_pipeline(
    generate: Callable[[int], Mapping[str, jnp.ndarray]],
    output_path: str,
    num_train: int,
    num_test: int,
    return_generated_examples: bool = False,
) -> Callable[[], PipelineOutput]:
  """Runs the generation pipeline for the HML datasets."""
  def pipeline() -> PipelineOutput:
    train_examples = list()
    test_examples = list()
    with open(f"{output_path}/features.txt", "w") as f:
      save_features(f, generate(0))
    with tf.io.TFRecordWriter(f"{output_path}/train.tfrecord") as writer:
      for i in range(num_train):
        example = generate(i)
        if return_generated_examples:
          train_examples.append(example)
        example = tf.train.Example(features=tf.train.Features(
            feature=encode_example(example)))
        writer.write(example.SerializeToString())
    with tf.io.TFRecordWriter(f"{output_path}/test.tfrecord") as writer:
      for i in range(num_test):
        example = generate(num_train + i)
        if return_generated_examples:
          test_examples.append(example)
        example = tf.train.Example(features=tf.train.Features(
            feature=encode_example(example)))
        writer.write(example.SerializeToString())
    if return_generated_examples:
      return tuple(train_examples), tuple(test_examples)
  return pipeline


def generate_full_dataset(
    folder: str,
    dataset: str,
    dt: float,
    num_steps: int,
    steps_per_dt: int,
    num_train: int,
    num_test: int,
    overwrite: bool,
    return_generated_examples: bool = False,
) -> PipelineOutput:
  """Runs the data generation."""
  dt_str = str(dt).replace(".", "_")
  folder = os.path.join(folder, dataset.lower() + f"_dt_{dt_str}")
  set_up_folder(folder, overwrite)

  cls, config = globals().get(dataset.upper())
  system = cls(**config())
  generate = functools.partial(
      generate_sample,
      system=system,
      dt=dt,
      num_steps=num_steps,
      steps_per_dt=steps_per_dt)
  pipeline = create_pipeline(
      generate, folder, num_train, num_test, return_generated_examples)
  return pipeline()


MASS_SPRING = (
    ideal_mass_spring.IdealMassSpring,
    lambda: dict(  #  pylint:disable=g-long-lambda
        k_range=utils.BoxRegion(2.0, 2.0),
        m_range=utils.BoxRegion(0.5, 0.5),
        radius_range=utils.BoxRegion(0.1, 1.0),
        uniform_annulus=False,
        randomize_canvas_location=False,
        randomize_x=False,
        num_colors=1,
    )
)


MASS_SPRING_COLORS = (
    ideal_mass_spring.IdealMassSpring,
    lambda: dict(  #  pylint:disable=g-long-lambda
        k_range=utils.BoxRegion(2.0, 2.0),
        m_range=utils.BoxRegion(0.2, 1.0),
        radius_range=utils.BoxRegion(0.1, 1.0),
        num_colors=6,
    )
)


MASS_SPRING_COLORS_FRICTION = (
    ideal_mass_spring.IdealMassSpring,
    lambda: dict(  #  pylint:disable=g-long-lambda
        k_range=utils.BoxRegion(2.0, 2.0),
        m_range=utils.BoxRegion(0.2, 1.0),
        radius_range=utils.BoxRegion(0.1, 1.0),
        num_colors=6,
        friction=0.05,
    ),
)


PENDULUM = (
    ideal_pendulum.IdealPendulum,
    lambda: dict(  #  pylint:disable=g-long-lambda
        m_range=utils.BoxRegion(0.5, 0.5),
        g_range=utils.BoxRegion(3.0, 3.0),
        l_range=utils.BoxRegion(1.0, 1.0),
        radius_range=utils.BoxRegion(1.3, 2.3),
        uniform_annulus=False,
        randomize_canvas_location=False,
        num_colors=1,
    )
)


PENDULUM_COLORS = (
    ideal_pendulum.IdealPendulum,
    lambda: dict(  #  pylint:disable=g-long-lambda
        m_range=utils.BoxRegion(0.5, 1.5),
        g_range=utils.BoxRegion(3.0, 4.0),
        l_range=utils.BoxRegion(.5, 1.0),
        radius_range=utils.BoxRegion(1.3, 2.3),
        num_colors=6,
    )
)


PENDULUM_COLORS_FRICTION = (
    ideal_pendulum.IdealPendulum,
    lambda: dict(  #  pylint:disable=g-long-lambda
        m_range=utils.BoxRegion(0.5, 1.5),
        g_range=utils.BoxRegion(3.0, 4.0),
        l_range=utils.BoxRegion(.5, 1.0),
        radius_range=utils.BoxRegion(1.3, 2.3),
        num_colors=6,
        friction=0.05,
    )
)


DOUBLE_PENDULUM = (
    ideal_double_pendulum.IdealDoublePendulum,
    lambda: dict(  #  pylint:disable=g-long-lambda
        m_range=utils.BoxRegion(0.5, 0.5),
        g_range=utils.BoxRegion(3.0, 3.0),
        l_range=utils.BoxRegion(1.0, 1.0),
        radius_range=utils.BoxRegion(1.3, 2.3),
        uniform_annulus=False,
        randomize_canvas_location=False,
        num_colors=2,
    )
)


DOUBLE_PENDULUM_COLORS = (
    ideal_double_pendulum.IdealDoublePendulum,
    lambda: dict(  #  pylint:disable=g-long-lambda
        m_range=utils.BoxRegion(0.4, 0.6),
        g_range=utils.BoxRegion(2.5, 4.0),
        l_range=utils.BoxRegion(0.75, 1.0),
        radius_range=utils.BoxRegion(1.0, 2.5),
        num_colors=6,
    )
)


DOUBLE_PENDULUM_COLORS_FRICTION = (
    ideal_double_pendulum.IdealDoublePendulum,
    lambda: dict(  #  pylint:disable=g-long-lambda
        m_range=utils.BoxRegion(0.4, 0.6),
        g_range=utils.BoxRegion(2.5, 4.0),
        l_range=utils.BoxRegion(0.75, 1.0),
        radius_range=utils.BoxRegion(1.0, 2.5),
        num_colors=6,
        friction=0.05
    ),
)


TWO_BODY = (
    n_body.TwoBodySystem,
    lambda: dict(  #  pylint:disable=g-long-lambda
        m_range=utils.BoxRegion(1.0, 1.0),
        g_range=utils.BoxRegion(1.0, 1.0),
        radius_range=utils.BoxRegion(0.5, 1.5),
        provided_canvas_bounds=utils.BoxRegion(-2.75, 2.75),
        randomize_canvas_location=False,
        num_colors=2,
    )
)


TWO_BODY_COLORS = (
    n_body.TwoBodySystem,
    lambda: dict(  #  pylint:disable=g-long-lambda
        m_range=utils.BoxRegion(0.5, 1.5),
        g_range=utils.BoxRegion(0.5, 1.5),
        radius_range=utils.BoxRegion(0.5, 1.5),
        provided_canvas_bounds=utils.BoxRegion(-5.0, 5.0),
        randomize_canvas_location=False,
        num_colors=6,
    )
)


def no_open_spiel_func(*_, **__):
  raise ValueError("You must download and install `open_spiel` first in "
                   "order to use the game_dynamics datasets. See "
                   "https://github.com/deepmind/open_spiel for instructions"
                   " how to do this.")

if not open_spiel_available():
  MATCHING_PENNIES = (no_open_spiel_func, dict)
  ROCK_PAPER_SCISSORS = (no_open_spiel_func, dict)
else:
  MATCHING_PENNIES = (game_dynamics.ZeroSumGame,
                      lambda: dict(game_name="matrix_mp"))
  ROCK_PAPER_SCISSORS = (game_dynamics.ZeroSumGame,
                         lambda: dict(game_name="matrix_rps"))
