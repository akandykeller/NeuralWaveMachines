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
"""A module for loading the Hamiltonian datasets."""
import functools
import os
from typing import Optional, Mapping, Any, Tuple, Sequence, Callable, Union, TypeVar

import jax
import tensorflow as tf
import tensorflow_datasets as tfds


T = TypeVar("T")


def filter_based_on_keys(
    example: Mapping[str, T],
    keys_to_preserve: Sequence[str],
    single_key_return_array: bool = False
) -> Union[T, Mapping[str, T]]:
  """Filters the contents of the mapping, to return only the keys given in `keys_to_preserve`."""
  if not keys_to_preserve:
    raise ValueError("You must provide at least one key to preserve.")
  if len(keys_to_preserve) == 1 and single_key_return_array:
    return example[keys_to_preserve[0]]
  elif single_key_return_array:
    raise ValueError(f"You have provided {len(keys_to_preserve)}>1 keys to "
                     f"preserve and have also set "
                     f"single_key_return_array=True.")
  return {k: example[k] for k in keys_to_preserve}


def preprocess_batch(
    batch: Mapping[str, Any],
    num_local_devices: int,
    multi_device: bool,
    sub_sample_length: Optional[int],
    dtype: str = "float32"
) -> Mapping[str, Any]:
  """Function to preprocess the data for a batch.

  This performs two functions:
    1.If 'sub_sample_length' is not None, it randomly subsamples every example
    along the second (time) axis to return an array of the requested length.
     Note that this assumes that all arrays have the same length.
    2. Converts all arrays to the provided data type using
      `tf.image.convert_image_dtype`.
    3. Reshapes the array based on the number of local devices.

  Args:
    batch: Dictionary with the full batch.
    num_local_devices: The number of local devices.
    multi_device: Whether to prepare the batch for multi device training.
    sub_sample_length: The sub-sampling length requested.
    dtype: String for the dtype, must be a floating-point type.

  Returns:
    The preprocessed batch.
  """
  if dtype not in ("float32", "float64"):
    raise ValueError("The provided dtype must be a floating point dtype.")
  tensor = batch.get("image", batch.get("x", None))
  if tensor is None:
    raise ValueError("We need either the key 'image' or 'x' to be present in "
                     "the batch provided.")
  if not isinstance(tensor, tf.Tensor):
    raise ValueError(f"Expecting the value for key 'image' or 'x' to be a "
                     f"tf.Tensor, instead got {type(tensor)}.")
  # `n` here represents the batch size
  n = tensor.shape[0] or tf.shape(tensor)[0]
  # `t` here represents number of time steps in the batch
  t = tensor.shape[1]
  if sub_sample_length is not None:
    # Sample random index for each example in the batch
    limit = t - sub_sample_length + 1
    start = tf.random.uniform(shape=[n], maxval=limit, dtype="int32")
    indices = tf.range(sub_sample_length)[None, :, None] + start[:, None, None]
    def index(x):
      """Indexes every array in the batch according to the sampled indices and length, if its second dimensions is equal to `t`."""
      if x.shape.rank > 1 and x.shape[1] == t:
        if isinstance(n, tf.Tensor):
          shape = [None, sub_sample_length] + list(x.shape[2:])
        else:
          shape = [n, sub_sample_length] + list(x.shape[2:])
        x = tf.gather_nd(x, indices, batch_dims=1)
        x.set_shape(shape)
      return x

    batch = jax.tree_map(index, batch)

  def convert_fn(x):
    """Converts the value of `x` to the provided precision dtype.

      Integer valued arrays, with data type different from int32 and int64 are
      assumed to represent compressed images and are converted via
      `tf.image.convert_image_dtype`. For any other data types (float or int)
      their type is preserved, but their precision is changed based on the
      target `dtype`. For instance, if `dtype=float32` the float64 variables are
      converted to float32 and int64 values are converted to int32.

    Args:
      x: The input array.

    Returns:
      The converted output array.
    """
    if x.dtype == tf.int64:
      return tf.cast(x, "int32") if dtype == "float32" else x
    elif x.dtype == tf.int32:
      return tf.cast(x, "int64") if dtype == "float64" else x
    elif x.dtype == tf.float64 or x.dtype == tf.float32:
      return tf.cast(x, dtype=dtype)
    else:
      return tf.image.convert_image_dtype(x, dtype=dtype)

  batch = jax.tree_map(convert_fn, batch)
  if not multi_device:
    return batch
  def reshape_for_jax_pmap(x):
    """Reshapes values such that their leading dimension is the number of local devices."""
    return tf.reshape(x, [num_local_devices, -1] + x.shape[1:].as_list())
  return jax.tree_map(reshape_for_jax_pmap, batch)


def load_filenames_and_parse_fn(
    path: str,
    tfrecord_prefix: str
) -> Tuple[Tuple[str], Callable[[str], Mapping[str, Any]]]:
  """Returns the file names and read_fn based on the number of shards."""
  file_name = os.path.join(path, f"{tfrecord_prefix}.tfrecord")
  if not os.path.exists(file_name):
    raise ValueError(f"The dataset file {file_name} does not exist.")
  features_file = os.path.join(path, "features.txt")
  if not os.path.exists(features_file):
    raise ValueError(f"The dataset features file {features_file} does not "
                     f"exist.")
  with open(features_file, "r") as f:
    dtype_dict = dict()
    shapes_dict = dict()
    parsing_description = dict()
    for line in f:
      key = line.split(", ")[0]
      shape_string = line.split("(")[1].split(")")[0]
      shapes_dict[key] = tuple(int(s) for s in shape_string.split(",") if s)
      dtype_dict[key] = line.split(", ")[-1][:-1]
      if dtype_dict[key] == "uint8":
        parsing_description[key] = tf.io.FixedLenFeature([], tf.string)
      elif dtype_dict[key] in ("float32", "float64"):
        parsing_description[key] = tf.io.VarLenFeature(tf.int64)
      else:
        parsing_description[key] = tf.io.VarLenFeature(dtype_dict[key])

  def parse_fn(example_proto: str) -> Mapping[str, Any]:
    raw = tf.io.parse_single_example(example_proto, parsing_description)
    parsed = dict()
    for name, dtype in dtype_dict.items():
      value = raw[name]
      if dtype == "uint8":
        value = tf.image.decode_png(value)
      else:
        value = tf.sparse.to_dense(value)
        if dtype in ("float32", "float64"):
          value = tf.bitcast(value, type=dtype)
      value = tf.reshape(value, shapes_dict[name])
      if "/" in name:
        k1, k2 = name.split("/")
        if k1 not in parsed:
          parsed[k1] = dict()
        parsed[k1][k2] = value
      else:
        parsed[name] = value
    return parsed

  return (file_name,), parse_fn


def load_parsed_dataset(
    path: str,
    tfrecord_prefix: str,
    num_shards: int,
    shard_index: Optional[int] = None,
    keys_to_preserve: Optional[Sequence[str]] = None
) -> tf.data.Dataset:
  """Loads a dataset and shards it based on jax devices."""
  shard_index = shard_index or jax.process_index()
  file_names, parse_fn = load_filenames_and_parse_fn(
      path=path,
      tfrecord_prefix=tfrecord_prefix,
  )

  ds = tf.data.TFRecordDataset(file_names)

  threads = max(1, os.cpu_count() - 4)
  options = tf.data.Options()
  options.threading.private_threadpool_size = threads
  options.threading.max_intra_op_parallelism = 1
  ds = ds.with_options(options)

  # Shard if we don't shard by files
  if num_shards != 1:
    ds = ds.shard(num_shards, shard_index)

  # Parse the examples one by one
  if keys_to_preserve is not None:
    # Optionally also filter them based on the keys provided
    def parse_filter(example_proto):
      example = parse_fn(example_proto)
      return filter_based_on_keys(example, keys_to_preserve=keys_to_preserve)
    ds = ds.map(parse_filter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  else:
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return ds


def load_dataset(
    path: str,
    tfrecord_prefix: str,
    sub_sample_length: Optional[int],
    per_device_batch_size: int,
    num_epochs: Optional[int],
    drop_remainder: bool,
    multi_device: bool = False,
    num_shards: int = 1,
    shard_index: Optional[int] = None,
    keys_to_preserve: Optional[Sequence[str]] = None,
    shuffle: bool = False,
    cache: bool = True,
    shuffle_buffer: Optional[int] = 10000,
    dtype: str = "float32",
    seed: Optional[int] = None
) -> tf.data.Dataset:
  """Creates a tensorflow.Dataset pipeline from an TFRecord dataset.

  Args:
    path: The path to the dataset.
    tfrecord_prefix: The dataset prefix.
    sub_sample_length: The length of the sequences that will be returned.
      If this is `None` the dataset will return full length sequences.
      If this is an `int` it will subsample each sequence uniformly at random
      for a sequence of the provided size. Note that all examples in the dataset
      must be at least this long, otherwise the tensorflow code might crash.
    per_device_batch_size: The batch size to use on a single device. The actual
      batch size is this multiplied by the number of devices.
    num_epochs: The number of times to repeat the full dataset.
    drop_remainder: If the number of examples in the dataset are not divisible
      evenly by the batch size, whether each epoch to drop the remaining
      examples, or to construct a batch with batch size smaller than usual.
    multi_device: Whether to load the dataset prepared for multi-device use
      (e.g. pmap) with leading dimension equal to the number of local devices.
    num_shards: If you want to shard the dataset, you must specify how many
      shards you want to use.
    shard_index: The shard index for this host. If `None` will use
      `jax.process_index()`.
    keys_to_preserve: Explicit specification which keys to keep from the dataset
    shuffle: Whether to shuffle examples in the dataset.
    cache: Whether to use cache in the tf.Dataset.
    shuffle_buffer: Size of the shuffling buffer.
    dtype: What data type to convert the data to.
    seed: Seed to pass to the loader.
  Returns:
    A tensorflow dataset object.
  """
  per_host_batch_size = per_device_batch_size * jax.local_device_count()
  # Preprocessing function
  batch_fn = functools.partial(
      preprocess_batch,
      num_local_devices=jax.local_device_count(),
      multi_device=multi_device,
      sub_sample_length=sub_sample_length,
      dtype=dtype)

  with tf.name_scope("dataset"):
    ds = load_parsed_dataset(
        path=path,
        tfrecord_prefix=tfrecord_prefix,
        num_shards=num_shards,
        shard_index=shard_index,
        keys_to_preserve=keys_to_preserve,
    )
    if cache:
      ds = ds.cache()
    if shuffle:
      ds = ds.shuffle(shuffle_buffer, seed=seed)
    ds = ds.repeat(num_epochs)
    ds = ds.batch(per_host_batch_size, drop_remainder=drop_remainder)
    ds = ds.map(batch_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds.prefetch(tf.data.experimental.AUTOTUNE)


def dataset_as_iter(dataset_func, *args, **kwargs):
  def iterable_func():
    yield from tfds.as_numpy(dataset_func(*args, **kwargs))
  return iterable_func
