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

"""Generates the samples from the models."""
import logging
import operator
import os
import re
import subprocess
import tempfile
import time
from typing import Dict

import absl.app as app
import absl.flags as flags
import absl.logging as absl_logging
import colored_traceback.auto
import rich
import rich.console
import rich.panel
import rich.style
import tensorflow as tf
import tqdm
import transformers


import constants
import task_specific
import tf_utils
import utils


LOGGER = logging.getLogger(__name__)
_ACCEPTABLE_APPROACHES = frozenset([
  constants.ApproachTypeChoices.naked_lm,
  constants.ApproachTypeChoices.cached_pretok
])


_FLAG_H5_MODEL_PATH = flags.DEFINE_string(
  "h5_path",
  None,
  "Path to the model save."
)
_FLAG_CKPT_MODEL_PATH = flags.DEFINE_string(
  "ckpt_path",
  None,
  "Path to the model save."
)
_FLAG_APPROACH_TYPE = flags.DEFINE_enum(
  "approach_type",
  None,
  _ACCEPTABLE_APPROACHES,
  "Path to the model save."
)
_FLAG_OUTPUT_PATH = flags.DEFINE_string(
  "output_path",
  None,
  "Where to save the generations. A json file. Can be on Google Cloud."
)

_FLAG_DATASET_TYPE = flags.DEFINE_enum(
  "dataset_type",
  "tfr",
  constants.DatasetTypeChoices.choices(),
  "Whether to use the hdf5 or the tfr pipeline."
)

# Need one here
_FLAG_TFR_PREFIX = flags.DEFINE_string(
  "tfr_prefix",
  None,
  "Glob prefix of the tfr files."
)

# 1 or 2 ?
_FLAG_BATCH_SIZE = flags.DEFINE_integer(
  "batch_size",
  None,
  "Size of the batch PER DEVICE."
)

# ok
_FLAG_SPLIT = flags.DEFINE_enum(
  "split",
  "test",
  {"eval", "test"},
  "Which split to generate from."
)


_FLAG_GENERATION_LENGTH_LIMIT = flags.DEFINE_integer(
  "generation_length_limit",
  None,
  "Number of tokens to reserve for generation at the end."
)

# No flag necessary
_FLAG_IS_LOCAL_TPU = flags.DEFINE_bool(
  "tpu-is-local",
  True,
  "Whether we are using a one-vm TPU.",
)

# No flag necessary
_FLAG_TPU_NAME = flags.DEFINE_string(
  "tpu-name",
  "",
  "Name of the TPU to use."
)

# No flag necessary
_FLAG_HF_MODEL_KEY = flags.DEFINE_string(
  "hf-model-key",
  "gpt2-xl",
  "Used when loading the model with checkpoints.",
)


def make_further_prep_generate_not_test(eos_token_id):
  def further_prep_generate_not_test(
      batch: Dict[str, tf.Tensor]
  ) -> tf.Tensor:
    """Removes padding tokens and answer tokens from the input text.
    """
    setup_tokens = batch["label_ids"] == -100
    text_tokens = batch["input_ids"] != eos_token_id
    context_tokens = tf.logical_and(setup_tokens, text_tokens)

    batch = tf.boolean_mask(batch["input_ids"], context_tokens)
    return batch
  return further_prep_generate_not_test


def make_further_prep_generate_test(eos_token_id):
  @tf.function
  def further_prep_generate_test(batch: Dict[str, tf.Tensor]) -> tf.Tensor:
    """Removes padding tokens from the input.
    For test, we don't have answers, so we can assume that the whole
    thing is setup text.

    # TODO: maybe merge with other version of this function.
    """

    text_tokens = batch["input_ids"] != eos_token_id
    batch = tf.boolean_mask(
      batch["input_ids"], text_tokens
    )
    return batch
  return further_prep_generate_test


def make_model_tf(path: str, mode: str) -> tf.Tensor:
  """Prepare the model for generation.
  Loads the model architecture from the huggingface pre-trained model, then
  loads a checkpoint.

  TODO: There must be a way to just load from config + checkpoint, no pretrained
    weights.
  """
  with utils.log_duration(LOGGER, make_model_tf.__name__, "Load model."):
    if mode == constants.SaveModeChoices.hfh5:
      config_path = os.path.join(path, "config.json")
      model_path = os.path.join(path, "tf_model.h5")
      utils.check_exists(config_path)
      utils.check_exists(model_path)
      config = transformers.GPT2Config.from_pretrained(config_path)
      return transformers.TFGPT2LMHeadModel.from_pretrained(
        model_path,
        config=config
      )
    elif mode == constants.SaveModeChoices.ckpt:
      model = transformers.TFGPT2LMHeadModel.from_pretrained(
        _FLAG_HF_MODEL_KEY.value,
      )
      ckpt = tf.train.Checkpoint(model=model)
      ckpt.restore(_FLAG_CKPT_MODEL_PATH.value)
    else:
      raise RuntimeError(f"Unsupported Save Mode: {mode}")
  return model


def make_print_sample():
  # Monokai
  title_color = "#6c99bb"
  normal_color = "#d6d6d6"
  background_color = "#2e2e2e"
  titles = ["Question:", "Answer:", "Context:"]

  def print_sample(sample, title, console):
    """Pretty print samples using Python rich.

    The parsing is pretty frail, but that's not a big deal.
    """
    # sample = sample.replace("\n", " <\\n> ")
    for title in titles:
      sample = sample.replace(
        title, f"\n\n[{title_color} bold]{title}[/]"
      )

    panel = rich.panel.Panel(
      sample.strip(),
      title=title,
      style=rich.style.Style(
        bgcolor=background_color, color=normal_color
      )
    )

    console.print(panel)
  return print_sample

def main(argv):
  if len(argv) > 1:
    raise RuntimeError(argv[1:])
  absl_logging.use_python_logging()
  utils.check_contained(_FLAG_APPROACH_TYPE.value, _ACCEPTABLE_APPROACHES)

  utils.check_operator(
    operator.xor,
    bool(_FLAG_H5_MODEL_PATH.value),
    bool(_FLAG_CKPT_MODEL_PATH.value)
  )

  if _FLAG_H5_MODEL_PATH.value:
    model_path = _FLAG_H5_MODEL_PATH.value
    mode = constants.SaveModeChoices.hfh5
  elif _FLAG_CKPT_MODEL_PATH.value:
    model_path = _FLAG_CKPT_MODEL_PATH.value
    mode = constants.SaveModeChoices.ckpt
  else:
    raise RuntimeError("Logically should never happen.")

  utils.check_exists(model_path)
  device_type = tf_utils.devices_to_use()[0].device_type

  # ONLY GPU IS SUPPORTED
  utils.check_equal(device_type, "GPU")


  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Build the distribution strategy
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if device_type == "TPU":
    # ONLY LOCAL TPU IS "SUPPORTED"
    utils.check_isinstance(_FLAG_IS_LOCAL_TPU.value, bool)
    assert _FLAG_IS_LOCAL_TPU.value
    tpu_config = tf_utils.init_tpus(local=True)
    utils.check_isinstance(tpu_config, tf_utils.TpuConfigType)
    utils.check_not_none(tpu_config)
    strategy = tf.distribute.TPUStrategy(tpu_config.resolver)
  elif device_type == "GPU":
    strategy = tf.distribute.MirroredStrategy(
      devices=tf.config.experimental.list_logical_devices('GPU')
    )
  else:
    raise RuntimeError(device_type)

  # ONLY GPU IS SUPPORTED
  print(tf.config.list_logical_devices())
  utils.check_isinstance(strategy, tf.distribute.MirroredStrategy)

  ##############################################################################
  # Load Model
  ##############################################################################
  with utils.log_duration(LOGGER, main.__name__, "All of model preparation"):
    with strategy.scope():
      # HF isn't able to read directly from GCS
      if (model_path.startswith("gs://")
          and mode == constants.SaveModeChoices.hfh5):
        with utils.log_duration(
            LOGGER,
            main.__name__,
            "Download model from GS"
        ):
          with tempfile.TemporaryDirectory() as td:
            td += os.path.sep

            if os.path.exists("/root/google-cloud-sdk/bin/gsutil"):
              exec_ = "/root/google-cloud-sdk/bin/gsutil"
            else:
              exec_ = "gsutil"

            command = [
              exec_,
              "-m",
              "cp",
              "-r",
              os.path.join(model_path, "*"),
              td,
            ]
            LOGGER.debug("Running bash command: %s", " ".join(command))
            subprocess.check_call(command)
            LOGGER.debug(
              "Files at the temp dir(%s): %s", td, str(os.listdir(td))
            )

            model = make_model_tf(td, mode=mode)
      else:
        model = make_model_tf(model_path, mode=mode)

  # model.__call__ = tf.function(
  #     model.__call__,
  #     experimental_relax_shapes=True,
  #     # experimental_compile=True,
  # )
  #
  # model.generate = tf.function(
  #     model.generate,
  #     experimental_relax_shapes=True,
  #     # experimental_compile=True,
  # )

  utils.check_not_none(model)

  ##############################################################################
  # Load Dataset Pipeline
  ##############################################################################
  utils.check_contained(_FLAG_APPROACH_TYPE.value, {
    constants.ApproachTypeChoices.naked_lm,
    constants.ApproachTypeChoices.cached_pretok
  })
  devices = tf_utils.devices_to_use()
  num_replicas = (
    len(devices) if devices[0].device_type in {"GPU", "TPU"} else 1
  )
  utils.check_equal(devices[0].device_type, "GPU")

  # Only a batch size of 1 is currently supported. We need attention masks
  utils.check_equal(_FLAG_BATCH_SIZE.value, 1)
  batch_size = _FLAG_BATCH_SIZE.value * num_replicas
  approach_type = _FLAG_APPROACH_TYPE.value

  logging.debug("Loading dataset.")
  tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2-xl")
  ds = task_specific.create_lm_ds_kilt_eli5(
    tokenizer=tokenizer,
    context_window_size=1024,
    dataset_name="kilt_eli5",
    batch_size=1,  # >> We set our own batch size elsewhere
    db_path=None, # None,
    random_seed=0,
    use_subset=False,
    subset_size=-1,
    use_helper_words=True,
    approach_type=approach_type,
    num_retrievals=5,  # Will never change
    retrieval_temperature=1.,
    retriever=None,  # Cached retrievals don't need a retriever
    repeat=False,  # Will never change
    split=_FLAG_SPLIT.value,
    enable_debug_checks=False,
    retrieval_bank_size=5,  # Will never change
    dataset_type=_FLAG_DATASET_TYPE.value,
    tfr_prefix=_FLAG_TFR_PREFIX.value,
    qty_shuffle=1,  # Will never change
    max_length_generation=350
  )

  if _FLAG_SPLIT.value == constants.SplitChoices.test:
    ds = ds.map(make_further_prep_generate_test(tokenizer.eos_token_id))
  else:
    ds = ds.map(make_further_prep_generate_not_test(tokenizer.eos_token_id))

  # Pad to the max length
  ds = ds.padded_batch(
    batch_size=batch_size, padding_values=tokenizer.eos_token_id
  )
  ds = strategy.experimental_distribute_dataset(ds)

  ##############################################################################
  # Generate
  ##############################################################################
  LOGGER.debug("Generating.")
  generations = []
  num_entries_in_split = (
    task_specific.DATASET_CARDINALITIES["kilt_eli5"][_FLAG_SPLIT.value]
  )

  entries_counter = tqdm.tqdm(total=num_entries_in_split)
  for batch_no, batch in enumerate(ds):
    # Calling model.generate. We should make a config file with the
    # hyperparameters for generation, or make a facility in the one we already
    # have. I feel like a separate one would be better, separating concerns.
    output = strategy.run(model.generate, kwargs=dict(
      input_ids=batch,
      max_length=_FLAG_GENERATION_LENGTH_LIMIT.value,
      use_cache=True,
      attention_mask=tf.cast(batch != tokenizer.eos_token_id, tf.int32),
      repetition_penalty=2.,
    ))
    output = tf_utils.process_strat_output(
      strategy_outputs=output,
      current_batch_size=batch_size,
      strategy=strategy,
      name="generations"
    )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display the inputs and outputs.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    rich_console = rich.console.Console(color_system="256")
    print_sample = make_print_sample()

    with utils.log_duration(
        LOGGER, "main", "all of tokenizer.decode for a batch."
    ):
      for i in range(batch_size):
        input_text = tokenizer.decode(batch.numpy()[i])
        output_text = tokenizer.decode(output.numpy()[i])
        print("#" * 1000)
        print(f"Batch {batch_no} Generation {i}")
        print_sample(
          input_text, f"input batch_no {batch_no}", rich_console
        )
        print_sample(
          output_text, f"output batch_no {batch_no}", rich_console
        )
        generations.append(output_text)
      print("#" * 1000)
    entries_counter.update(batch.shape[0])

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Save the output to a JSON File.
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  utils.to_json_file(
    os.path.join(
      _FLAG_OUTPUT_PATH.value,
      _FLAG_SPLIT.value,
      _FLAG_APPROACH_TYPE.value,
      time.strftime("%Y%m%d-%H%M%S.json")
    ),
    dict(
      flags={
        flag.name: flag.value
        for flag in flags.FLAGS.flags_by_module_dict()[argv[0]]
      },
      generations=generations
    )
  )
  logging.debug("Saved to: %s", _FLAG_OUTPUT_PATH.value)

if __name__ == "__main__":
  app.run(main)
