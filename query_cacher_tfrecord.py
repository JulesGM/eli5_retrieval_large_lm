# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Iterates over the whole ELI5 dataset (for each dataset split), extracts REALM or REALM++ embeddings, then does exact retrieval for a large number of neighbors, then saves the resulting db indices and distance (inner product).

This makes it so we don't have to do real retrieval during training, we just
sample from the neighbors as a function of the inner product, which is much
faster.

Examples of use:

# Remote use
pytype query_cacher_tfrecord.py -P . --check-variable-types \
  --check-container-types \
  --check-parameter-types --precise-return && \
  python3 check_flags.py query_cacher_tfrecord.py && \
  FLAGS=$(python3 json_to_args.py configs/query_cacher_tfr_configs/remote.json) && \
  python3 query_cacher_tfrecord.py $FLAGS

  """
import collections
import logging
import multiprocessing
import os
import resource
import time
from typing import List, Callable, Dict

from absl import app
from absl import flags
from absl import logging as absl_logging
import bert_utils
import colored_traceback.auto  # pylint: disable=unused-import
import constants
import datasets
import numpy as np
import tensorflow as tf

major, minor, _ = list(map(int, tf.__version__.strip().split(".")))
assert major == 2, tf.__version__
assert minor == 4, tf.__version__

import tensorflow.python.framework.ops as ops
import tensorflow_hub as hub
import tf_utils
import tqdm
import transformers
import utils


# Minimum number of file descriptors to be set with `ulimit -n`. Corresponds
# to the default value given by `ulimit -n` on a ubuntu desktop. We don't
# call `ulimit -n` dynamically to fetch the value in order to not accidentally
# create a chain loop of ever increasing `ulimit -n`, even if it is super
# unlikely.
_MIN_N_FD = 256

_FLAG_TPU_NAME = flags.DEFINE_string(
  "tpu_name",
  None,
  "Name of the TPU",
)

_FLAG_OUTPUT_PATH = flags.DEFINE_string(
    "output_dir",
    None,
    "Directory in which to save, on the cloud."
)
_FLAG_RETRIEVER_CONFIG_PATH = flags.DEFINE_string(
    "retriever_config_path",
    None,
    "Path to the retriever's configuration file."
)
_FLAG_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    100,
    "Size of the batch for the encoder BERT model."
)

_FLAG_DATASET_ROOT = flags.DEFINE_string(
    "dataset_root",
    None,
    "Root of the place where the datasets are saved."
)

# Flags specific to query encoding
_FLAG_EMBEDDING_DEPTH = flags.DEFINE_integer(
    "embedding_depth",
    128,
    "Size of the BERT (REALM) embeddings.",
)

# Flags specific to retrieval caching
_FLAG_NUM_RETRIEVALS = flags.DEFINE_integer(
    "num_retrievals",
    10,
    "Number of neighbors to retrieve.",
)
_FLAG_CONTEXT_SIZE = flags.DEFINE_integer(
    "context_size",
    1024,
    "Length to pad to."
)
_FLAG_MAX_LENGTH_RETRIEVALS = flags.DEFINE_integer(
    "max_length_retrievals",
    350,
    "Maximum length of the retrievals."
)

_FLAG_NUM_SHARDS = flags.DEFINE_integer(
    "num_shards",
    2048,
    "Number of files to output tfr shards."
)

_FLAG_USE_SUBSET = flags.DEFINE_boolean(
    "use_subset",
    False,
    "Whether or not to use a subset."
)

_FLAG_SUBSET_QTY = flags.DEFINE_integer(
    "subset_qty",
    500,
    "When using a subset of the dataset to debug this script, how big should "
    "that dataset be."
)

_FLAG_TPU_IS_LOCAL = flags.DEFINE_boolean(
  "tpu_is_local",
  True,
  "Whether the"
)

LOGGER = logging.getLogger(__name__)




def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class BERTBatchFields(utils.FlagChoices):
  bert_question_token_ids = "bert_question_token_ids"
  bert_attention_masks = "bert_attention_masks"


def _make_transform_fn(
    bert_tokenizer,
    bert_cls_token_id,
    bert_sep_token_id,
):
  """Prepares the transformation function."""
  @tf.function
  def _prepare_for_bert(sample):
    """Prepares a question sample from ELI5 to be fed to BERT."""
    bert_question_token_ids = bert_tokenizer.tokenize(
        tf.expand_dims(sample["question"], 0))
    bert_question_token_ids = tf.cast(
        bert_question_token_ids.merge_dims(1, 2).to_tensor(), tf.int32)
    cls_ids = tf.fill([tf.shape(bert_question_token_ids)[0], 1],
                      bert_cls_token_id)
    sep_ids = tf.fill([tf.shape(bert_question_token_ids)[0], 1],
                      bert_sep_token_id)
    bert_question_token_ids = tf.concat(
        (cls_ids, bert_question_token_ids, sep_ids), 1)

    return dict(
        bert_question_token_ids=bert_question_token_ids,
        bert_attention_masks=tf.ones_like(bert_question_token_ids),
        **sample
    )

  return _prepare_for_bert


@tf.function
def _squeeze(batch):
  """Squeezes and converts tensors to dense tensors w/ padding."""
  batch = dict(**batch)
  batch[BERTBatchFields.bert_question_token_ids] = tf.squeeze(
      batch[BERTBatchFields.bert_question_token_ids].to_tensor(0), 1)
  batch[BERTBatchFields.bert_attention_masks] = tf.squeeze(
      batch[BERTBatchFields.bert_attention_masks].to_tensor(0), 1)
  return batch


def _make_encode_fn(
    query_encoder
):
  """Prepares the BERT encoder function."""

  @tf.function(experimental_relax_shapes=True)
  def _encode(batch):
    """Encodes a sample with REALM BERT."""
    # Add a CLS token at the start of the input, and a SEP token at the end

    return query_encoder.signatures["projected"](
        input_ids=batch[BERTBatchFields.bert_question_token_ids],
        input_mask=batch[BERTBatchFields.bert_attention_masks],
        segment_ids=tf.zeros_like(
            batch[BERTBatchFields.bert_question_token_ids]
        ))["default"]

  return _encode


def make_encode_fn_strategy_run_fn(
    strategy,
    encode_fn,
):
  """Builds the runner function for the REALM query function."""

  # Giving {} as a default value would make the default value mutable, which
  # is prohibited (because changing the object would change the default value).

  @tf.function(experimental_relax_shapes=True)
  def encode_fn_strategy_run_fn(batch):
    """Runs the distribute strategy on the query encoder."""
    return strategy.run(encode_fn, args=(batch,))

  return encode_fn_strategy_run_fn


######################################################################
# Effectuate the retrievals.
######################################################################
def _prep_field(field, gpt2_tokenizer):
  """Prepares different fields to be saved in a tfr."""
  decoded_list = [sample.decode() for sample in field.numpy().tolist()]
  encoded = gpt2_tokenizer.batch_encode_plus(
      decoded_list,
      padding="max_length",
      truncation=True,
  ).input_ids

  ids = np.array(
      encoded,
      dtype=np.int32,
  )

  ids[ids == gpt2_tokenizer.eos_token_id] = -1
  return ids


def main(argv):
  # Arguments and logging boilerplate
  if len(argv) > 1:
    raise RuntimeError(argv)

  absl_logging.use_python_logging()
  utils.log_module_args(LOGGER, argv[0])

  # Load a retriever config.
  retriever_config = tf_utils.REALMConfig(
      **utils.from_json_file(_FLAG_RETRIEVER_CONFIG_PATH.value)
  )
  assert not _FLAG_USE_SUBSET.value

  # Preparation of the output path
  time_stamp = time.strftime("%Y%m%d-%H%M%S")
  target_path = os.path.join(_FLAG_OUTPUT_PATH.value, time_stamp.strip())
  if target_path[-1] != "/":
    target_path += "/"

  ##############################################################################
  # Setup devices and strategy
  ##############################################################################
  # Duration is pretty much instantaneous
  with utils.log_duration(LOGGER, "main", "Initializing devices"):
    tpu_config = tf_utils.init_tpus(
      local=_FLAG_TPU_IS_LOCAL.value, tpu_name=_FLAG_TPU_NAME.value
    )
    device_type = tf_utils.current_accelerator_type()
    LOGGER.debug("Devices: %s", str(tf_utils.devices_to_use()))
    if _FLAG_TPU_NAME.value and device_type == "CPU":
      raise RuntimeError("Device is CPU and we expected a TPU.")

    if device_type == "TPU":
      if tpu_config is None:
        raise RuntimeError("We should have a tpu_config.")
      strategy = tf.distribute.TPUStrategy(tpu_config.resolver)
      batch_size = len(tf_utils.devices_to_use()) * _FLAG_BATCH_SIZE.value
    elif device_type == "GPU" or device_type == "CPU":
      strategy = tf.distribute.MirroredStrategy()
      batch_size = len(tf_utils.devices_to_use()) * _FLAG_BATCH_SIZE.value
    else:
      raise RuntimeError(device_type)

  ##############################################################################
  # Load the KILT ELI5 dataset.
  ##############################################################################
  # Takes a while
  eli5 = {}
  keys = ["train", "validation", "test"]
  gpt2_tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2-xl")
  gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

  with utils.log_duration(LOGGER, "main", "Loading the ELI5 datasets."):
    if _FLAG_DATASET_ROOT.value:
      for split in tqdm.tqdm(keys):
        load_path = os.path.join(
            _FLAG_DATASET_ROOT.value,
            "HuggingfaceDatasets",
            f"{split}_kilt_eli5.hf"
        )
        with tf.device("/job:localhost"):
          eli5[split] = datasets.load_from_disk(load_path)
    else:
      eli5 = datasets.load_dataset("kilt_tasks", "eli5")

  ##############################################################################
  # Load the dataset of the text that will be retrieved.
  ##############################################################################
  # Takes a long time
  with utils.log_duration(
      LOGGER, "Main", "Load the textual dataset"
  ):
    # Extract the appropriate text
    # The buffer_size is taken from the original ORQA code.
    blocks_dataset = tf.data.TFRecordDataset(
        retriever_config.text_records, buffer_size=512 * 1024 * 1024
    )
    blocks_dataset = blocks_dataset.batch(
        retriever_config.num_block_records, drop_remainder=False
    )
    blocks: tf.Tensor = tf.data.experimental.get_single_element(blocks_dataset)

  ############################################################################
  # Increase the number of maximum open file descriptors to make space
  # for all the shards.
  ############################################################################
  max_num_fd = _FLAG_NUM_SHARDS.value * 3 + _MIN_N_FD
  resource.setrlimit(resource.RLIMIT_NOFILE, (max_num_fd, max_num_fd))

  ############################################################################
  # Prepare the output files.
  ############################################################################
  writers = {}
  all_paths = {}

  for split in keys:
    maybe_subset = "_subset" if _FLAG_USE_SUBSET.value else ""
    # Prepare paths. They can't be in a generator. A function generator would be
    # fine though.
    paths = [os.path.join(target_path + maybe_subset, f"{split}_{i}.tfr")
             for i in range(_FLAG_NUM_SHARDS.value)]
    all_paths[split] = paths
    writers[split] = []

    # Create The TFR writers.
    for i, path in enumerate(paths):
      writers[split].append(tf.io.TFRecordWriter(path))

  # Load the reference DB. We used to accidentally do this once per split :O
  with utils.log_duration(LOGGER, "main", "Loading the reference db."):
    checkpoint_path = os.path.join(
        retriever_config.query_embedder_path,
        "encoded", "encoded.ckpt"
    )
    reference_db_device = tf_utils.device_mapping().CPUs[0].name
    with tf.device(reference_db_device):
      reference_db = tf_utils.load_reference_db(
          checkpoint_path,
          variable_name="block_emb",
      )

  ############################################################################
  # Prep the encoder and the tokenizer
  ############################################################################
  with utils.log_duration(
      LOGGER, "main", "Loading the encoder model and the tokenizer."
  ):
    with strategy.scope():
      query_encoder = hub.load(retriever_config.query_embedder_path, tags={})
    encode_fn = _make_encode_fn(query_encoder)
    encode_fn_strategy_run = make_encode_fn_strategy_run_fn(
        strategy=strategy,
        encode_fn=encode_fn,
        )

    vocab_file = os.path.join(
        retriever_config.query_embedder_path, "assets", "vocab.txt"
    )
    utils.check_exists(vocab_file)
    do_lower_case = query_encoder.signatures["tokenization_info"](
    )["do_lower_case"]
    tokenization_info = dict(
        vocab_file=vocab_file, do_lower_case=do_lower_case
    )

    tokenizer, vocab_lookup_table = bert_utils.get_tf_tokenizer(
        query_encoder, tokenization_info
    )

  ############################################################################
  # Preprocess the dataset
  ############################################################################
  cls_token_id = tf.cast(
      vocab_lookup_table.lookup(tf.constant("[CLS]")), tf.int32
  )
  sep_token_id = tf.cast(
      vocab_lookup_table.lookup(tf.constant("[SEP]")), tf.int32
  )
  transform = _make_transform_fn(
      bert_tokenizer=tokenizer,
      bert_cls_token_id=cls_token_id,
      bert_sep_token_id=sep_token_id,
  )

  feature_dtypes = {
      constants.CTH5Fields.distances:
          tf.float32,
      constants.CTH5Fields.gpt2_retrieved_ids:
          tf.int32,
      constants.CTH5Fields.gpt2_answer_ids_inputs:
          tf.int32,
      constants.CTH5Fields.gpt2_question_ids_inputs:
          tf.int32,
  }

  with utils.log_duration(LOGGER, "main", "generating codes"):
    for split in keys:
      sample_count = 0
      eli5: Dict[str, datasets.Dataset]

      if split != "test":
        for_slices = dict(
            sample_id=eli5[split]["id"],
            question=eli5[split]["input"],
            answer=[
              sample[0]["answer"] for sample in eli5[split]["output"]
            ]
        )
      else:
        for_slices = dict(
            sample_id=eli5[split]["id"],
            question=eli5[split]["input"],
        )

      ds = tf.data.Dataset.from_tensor_slices(for_slices)
      ds = ds.map(transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)

      ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(batch_size))
      ds = ds.map(_squeeze, num_parallel_calls=tf.data.experimental.AUTOTUNE)

      tqdm_inner = tqdm.tqdm(
          enumerate(ds),
          total=len(eli5[split]["id"]) // _FLAG_BATCH_SIZE.value,
          desc=f"Split `{split}`: Batches"
      )

      for i, batch in tqdm_inner:
        features = collections.defaultdict(list)

        ######################################################################
        # Enforce the current real batch size
        ######################################################################
        current_batch_size = batch["sample_id"].shape[0]
        for k, v in batch.items():
          utils.check_equal(v.shape[0], current_batch_size)
        ######################################################################

        gpt2_question_ids_inputs = _prep_field(
            batch["question"], gpt2_tokenizer
        )
        utils.check_equal(gpt2_question_ids_inputs.dtype, np.int32)
        utils.check_equal(
            gpt2_question_ids_inputs.shape[0], current_batch_size
        )

        if split != "test":
          gpt2_answer_ids_inputs = _prep_field(
              batch["answer"], gpt2_tokenizer
          )
          utils.check_equal(gpt2_answer_ids_inputs.dtype, np.int32)
          utils.check_equal(
              gpt2_answer_ids_inputs.shape[0], current_batch_size
          )

          assert len(gpt2_answer_ids_inputs.shape) == 2, (
              gpt2_answer_ids_inputs.shape
          )

        ######################################################################
        # Save the gpt2 tokenized question and answer
        ######################################################################

        features[constants.CTH5Fields.gpt2_question_ids_inputs].extend(
            gpt2_question_ids_inputs)

        if split != "test":
          features[constants.CTH5Fields.gpt2_answer_ids_inputs].extend(
              gpt2_answer_ids_inputs)

        ######################################################################
        # Encode the samples.
        ######################################################################
        batch = strategy.experimental_distribute_values_from_function(
            tf_utils.make_dict_distribute_fn(batch)
        )

        embeddings = encode_fn_strategy_run(batch)
        embeddings = tf_utils.process_strat_output(
            embeddings, "embeddings", strategy, current_batch_size
        )
        utils.check_isinstance(embeddings, ops.EagerTensor)
        utils.check_equal(embeddings.shape[0], current_batch_size)

        # pytype doesn't seem to see that we check the type
        utils.check_equal(embeddings.shape[1], _FLAG_EMBEDDING_DEPTH.value)  # pytype: disable=attribute-error

        ######################################################################
        # Retrieve.
        ######################################################################
        # Do exact retrieval
        with tf.device(reference_db_device):
          top_k, inner_prods = tf_utils.mips_exact_search(
              embeddings, _FLAG_NUM_RETRIEVALS.value, reference_db
          )

        # Collate the results
        top_k = tf_utils.process_strat_output(
            top_k, "top_k", strategy, current_batch_size
        )

        # Check the shapes
        utils.check_equal(
            inner_prods.shape, (current_batch_size, _FLAG_NUM_RETRIEVALS.value)
        )
        utils.check_equal(
            top_k.shape, (current_batch_size, _FLAG_NUM_RETRIEVALS.value)
        )

        # Save the distances
        features[constants.CTH5Fields.distances].extend(inner_prods)

        # Retrieve the text fields associated to the indices
        gathered = tf.gather(blocks, top_k).numpy()
        utils.check_equal(gathered.shape[0], current_batch_size)
        utils.check_equal(gathered.shape[1], _FLAG_NUM_RETRIEVALS.value)

        retrievals = []
        for index_in_batch in range(current_batch_size):
          # Put the appropriate byte strings in a list
          local_gathered = gathered[index_in_batch].tolist()
          utils.check_equal(len(local_gathered), _FLAG_NUM_RETRIEVALS.value)
          # Decode to utf-8
          local_gathered = [sample.decode() for sample in local_gathered]
          # Encode to GPT2 BPE
          token_ids = np.array(
              gpt2_tokenizer.batch_encode_plus(
                  local_gathered,
                  padding="max_length",
                  truncation=True,
              ).input_ids
          )

          # Make sure no line is empty
          # TODO(julesgm): Maybe optional
          for line in token_ids:
            assert not np.all(line == 0), line

          # Convert the eos_tokens
          token_ids[token_ids == gpt2_tokenizer.eos_token_id] = -1

          # Save the retrievals
          retrievals.append(token_ids)

        # Save the feature
        features[constants.CTH5Fields.gpt2_retrieved_ids] = retrievals

        utils.check_equal(
            retrievals[0].shape,
            (_FLAG_NUM_RETRIEVALS.value, _FLAG_CONTEXT_SIZE.value)
        )

        for k, v in features.items():
          utils.check_equal(len(v), current_batch_size)

        for index_in_batch in range(current_batch_size):
          feature_dict = {}
          for feature_k, feature_v in features.items():
            # Cast the feature to its appropriate dtype
            casted_feats = tf.cast(
              feature_v[index_in_batch], feature_dtypes[feature_k]
            )
            # Serialize the tensor to bytes
            feature_bytes = tf.io.serialize_tensor(casted_feats)
            # Build a bytes list tf.train.Feature object,
            # the serialization tree node
            feature_dict[feature_k] = _bytes_feature(feature_bytes)

          # Create the serialization tree root
          # Expects a list of features
          feature = tf.train.Features(feature=feature_dict)
          # Expects a tf.train.Features object
          example_obj = tf.train.Example(features=feature)

          # Serialize that to bytes
          serialized_example = example_obj.SerializeToString()

          # Write the bytes
          # TODO(julesgm): Parallelize this with a thread or a process pool &
          #   futures.
          writers[split][sample_count % _FLAG_NUM_SHARDS.value].write(
              serialized_example
          )
          sample_count += 1

        if sample_count % 1000 == 0:
          LOGGER.debug("Paths: %s", str(all_paths[split][0]))

      LOGGER.debug("Flushing and closing the `%s` writers", split)
      for writer in tqdm.tqdm(writers[split]):
        writer.flush()
        writer.close()

  LOGGER.debug("Done.")

if __name__ == "__main__":
  app.run(main)
