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
"""The training script for the HGN models."""
import functools

from absl import app
from absl import flags
from absl import logging
from HGNExperiment import HGNExperiment
from jaxline import platform

# Allow TF and Jax to coexist in memory without TF allocating all of it.
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
  flags.mark_flag_as_required("config")
  logging.set_stderrthreshold(logging.INFO)
  app.run(functools.partial(platform.main, HGNExperiment))
  # app.run(functools.partial(platform.main, _CreateNamedOnMain))