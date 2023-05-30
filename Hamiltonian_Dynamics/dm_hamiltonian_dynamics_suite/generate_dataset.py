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
"""Script to generate datasets from the configs in datasets.py."""
from absl import app
from absl import flags

from dm_hamiltonian_dynamics_suite import datasets
from dm_hamiltonian_dynamics_suite.molecular_dynamics import generate_dataset

import jax

flags.DEFINE_string("folder", None,
                    "The folder where to store the datasets.")
flags.DEFINE_string("dataset", None,
                    "The dataset from datasets.py to use or "
                    "'molecular_dynamics' respectively.")
flags.DEFINE_string("lammps_file", None,
                    "For dataset='molecular_dynamics' this should be the "
                    "LAMMPS trajectory file containing a sequence of timesteps "
                    "obtained from a MD simulation.")
flags.DEFINE_float("dt", None, "The delta time between two observations.")
flags.DEFINE_integer("num_steps", None, "The number of steps to simulate.")
flags.DEFINE_integer("steps_per_dt", 10,
                     "How many internal steps to do per a single observation "
                     "step.")
flags.DEFINE_integer("num_train", None,
                     "The number of training examples to generate.")
flags.DEFINE_integer("num_test", None,
                     "The number of test examples to generate.")
flags.DEFINE_boolean("overwrite", False, "Overwrites previous data.")

flags.mark_flag_as_required("folder")
flags.mark_flag_as_required("dataset")
flags.mark_flag_as_required("dt")
flags.mark_flag_as_required("num_steps")
flags.mark_flag_as_required("num_train")
flags.mark_flag_as_required("num_test")
FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise ValueError(f"Unexpected args: {argv[1:]}")
  if FLAGS.dataset == "molecular_dynamics":
    generate_dataset.generate_lammps_dataset(
        lammps_file=FLAGS.lammps_file,
        folder=FLAGS.output_path,
        dt=FLAGS.dt,
        num_steps=FLAGS.num_steps,
        num_train=FLAGS.num_train,
        num_test=FLAGS.num_test,
        shuffle=FLAGS.shuffle,
        seed=FLAGS.seed,
        overwrite=FLAGS.overwrite,
    )
  else:
    datasets.generate_full_dataset(
        folder=FLAGS.folder,
        dataset=FLAGS.dataset,
        dt=FLAGS.dt,
        num_steps=FLAGS.num_steps,
        steps_per_dt=FLAGS.steps_per_dt,
        num_train=FLAGS.num_train,
        num_test=FLAGS.num_test,
        overwrite=FLAGS.overwrite
    )


if __name__ == "__main__":
  jax.config.update("jax_enable_x64", True)
  app.run(main)
