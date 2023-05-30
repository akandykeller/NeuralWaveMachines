#!/bin/bash
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

if [[ "$#" -eq 1 ]]; then
  readonly DATASET="${1}"
else
   echo "Illegal number of parameters. Expected only the name of the dataset."
   exit 2
fi

readonly FOLDER="/tmp/hamiltonian_ml/datasets"
readonly DTS=(0.05 0.1)
readonly NUM_STEPS=255
readonly NUM_TRAIN=500
readonly NUM_TEST=200

for DT in "${DTS[@]}"; do
  python3 generate_dataset.py \
      --folder=${FOLDER} \
      --dataset="${DATASET}" \
      --dt="${DT}" \
      --num_steps=${NUM_STEPS} \
      --num_train=${NUM_TRAIN} \
      --num_test=${NUM_TEST} \
      --overwrite=true
done
