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
"""Module for testing the generation and loading of datasets."""
import os

from absl.testing import absltest
from absl.testing import parameterized

from dm_hamiltonian_dynamics_suite import datasets
from dm_hamiltonian_dynamics_suite import load_datasets

import jax
from jax import numpy as jnp
import tensorflow as tf


DATASETS_TO_TEST = [
    "mass_spring",
    "pendulum_colors",
    "double_pendulum_colors",
    "two_body_colors",
]

if datasets.open_spiel_available():
  DATASETS_TO_TEST += [
      "matching_pennies",
      "rock_paper_scissors",
  ]


class TestToyDataset(parameterized.TestCase):
  """Test class for the functions in `tag_graph_matcher.py`."""

  def compare_structures_all_the_same(self, example, batched_example):
    """Compares that the two examples are identical in structure and value."""
    self.assertEqual(
        jax.tree_structure(example),
        jax.tree_structure(batched_example),
        "Structures should be the same."
    )
    # The real example image is not converted however
    example["image"] = tf.image.convert_image_dtype(
        example["image"], dtype=batched_example["image"].dtype).numpy()
    for v1, v2 in zip(jax.tree_leaves(example),
                      jax.tree_leaves(batched_example)):
      self.assertEqual(v1.dtype, v2.dtype, "Dtypes should be the same.")
      self.assertEqual((1,) + v1.shape, v2.shape, "Shapes should be the same.")
      self.assertTrue(jnp.allclose(v1, v2[0]), "Values should be the same.")

  @parameterized.parameters(DATASETS_TO_TEST)
  def test_dataset(
      self,
      dataset,
      folder: str = "/tmp/dm_hamiltonian_dynamics_suite/tests/",
      dt: float = 0.1,
      num_steps: int = 100,
      steps_per_dt: int = 10,
      num_train: int = 10,
      num_test: int = 10,
  ):
    """Checks that the dataset generation and loading are working correctly."""

    # Generate the dataset
    train_examples, test_examples = datasets.generate_full_dataset(
        folder=folder,
        dataset=dataset,
        dt=dt,
        num_steps=num_steps,
        steps_per_dt=steps_per_dt,
        num_train=num_train,
        num_test=num_test,
        overwrite=True,
        return_generated_examples=True,
    )

    # Load train dataset
    dataset_path = dataset.lower() + "_dt_" + str(dt).replace(".", "_")
    ds = load_datasets.dataset_as_iter(
        load_datasets.load_dataset,
        path=os.path.join(folder, dataset_path),
        tfrecord_prefix="train",
        sub_sample_length=None,
        per_device_batch_size=1,
        num_epochs=1,
        drop_remainder=False,
        dtype="float64"
    )
    examples = tuple(x for x in ds())
    self.assertEqual(
        len(train_examples), len(examples),
        "Number of training examples not the same."
    )
    # Compare individual examples
    for example_1, example_2 in zip(train_examples, examples):
      self.compare_structures_all_the_same(example_1, example_2)

    # Load test dataset
    ds = load_datasets.dataset_as_iter(
        load_datasets.load_dataset,
        path=os.path.join(folder, dataset_path),
        tfrecord_prefix="test",
        sub_sample_length=None,
        per_device_batch_size=1,
        num_epochs=1,
        drop_remainder=False,
        dtype="float64"
    )

    examples = tuple(x for x in ds())
    self.assertEqual(
        len(test_examples), len(examples),
        "Number of test examples not the same."
    )
    # Compare individual examples
    for example_1, example_2 in zip(test_examples, examples):
      self.compare_structures_all_the_same(example_1, example_2)


if __name__ == "__main__":
  jax.config.update("jax_enable_x64", True)
  absltest.main()
