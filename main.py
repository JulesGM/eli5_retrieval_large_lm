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

""" Training script for the retrieval solution.
"""

import functools
import itertools
import json
import logging
import operator
import os
import shlex
import socket
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional

from absl import app
from absl import flags
from absl import logging as absl_logging
import colorama
import constants
import numpy as np
from rich import console
from rich import table
import task_specific
import tensor2tensor.utils.adafactor
import tensorflow as tf
import tensorflow.python.distribute.values as values
import tensorflow.python.framework.ops as ops
import tf_utils
import toolz
import transformers
import utils



os.environ["TOKENIZERS_PARALLELISM"] = "true"
LOGGER = logging.getLogger(__name__)
SCRIPT_DIRECTORY = os.path.realpath(os.path.dirname(__file__))

LOGGER.debug(
    "############################################################"
    ">>>>>>>>>>>>>>> Tensorflow version: %s <<<<<<<<<<<<<<<<"
    "############################################################",
    str(tf.__version__)
)

################################################################################
# Flag Definitions
################################################################################
FLAGS = flags.FLAGS

# It is now recommended that one uses the return values of DEFINE_* calls
# because they can by pytype-checked and the intellisense/linter can know
# if the wrong variable name is called, contrarily to the FLAGS.* case.

FLAG_ALPHA_MODE = flags.DEFINE_bool(
  "alpha_mode",
  False,
  "",
)
FLAG_APPROACH_TYPE = flags.DEFINE_enum(
    "approach_type",
    None,
    constants.ApproachTypeChoices.choices(),
    "Type of approach to use.\n"
)
FLAG_MODEL_KEY = flags.DEFINE_string(
    "model_key",
    None,
    "Hugging Face key associated to the pre-trained model."
)
FLAG_RUN_NAME = flags.DEFINE_string(
    "run_name",
    None,
    "Name of the run. Can be anything."
)
FLAG_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Where to save the results to."
)
FLAG_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    None,
    "Inference batch size."
)
FLAG_BATCH_SPLIT = flags.DEFINE_integer(
    "batch_split",
    None,
    "Used for manual_improved. Sub-batch size."
)
FLAG_TASK = flags.DEFINE_enum(
    "task",
    constants.TaskChoices.train,
    constants.TaskChoices.choices(),
    "Whether to train or to evaluate the mode."
)
FLAG_RANDOM_SEED = flags.DEFINE_integer(
    "random_seed", 0,
    "Random seed used used for the random elements of the script."
)
FLAG_DB_PATH = flags.DEFINE_string(
    "db_path",
    None,
    "Path to the h5 file containing the dataset prepared with query_cacher.py"
)

# TPU Specific Args
FLAG_EXPERIMENTAL_COMPILE = flags.DEFINE_bool(
    "experimental_compile",
    False,
    "Whether to use experimental compile with the train and eval functions."
)
FLAG_DISTRIBUTE_MODE = flags.DEFINE_enum(
    "distribute_mode",
    constants.DistributeModeChoices.onedevicestrategy,
    constants.DistributeModeChoices.choices(),
    "What type of infrastructure to use to distribute the work."
)
FLAG_NUM_REPLICAS = flags.DEFINE_integer(
    "num_replicas",
    1,
    "Number of replicas to use fordata parallelism."
)

# Training specific flags
FLAG_MODEL_OUTPUT_PATH = flags.DEFINE_string(
    "model_output_path",
    None,
    "Where to save the model."
)
FLAG_LEARNING_RATE = flags.DEFINE_float(
    "learning_rate",
    None,
    "Learning rate for the optimizer."
)
FLAG_BATCHES_BETWEEN_EVALS = flags.DEFINE_integer(
    "batches_between_evals",
    5,
    "Number of batches between eval passes."
)
FLAG_NUMBER_EVAL_BATCHES = flags.DEFINE_integer(
    "number_eval_batches",
    1,
    "Number of eval batches when doing an eval pass."
)
FLAG_USE_HELPER_WORDS = flags.DEFINE_boolean(
    "use_helper_words",
    True,
    "Whether to add guiding words in the inputs, like `Question:`,"
    " `Answer:` and `Context:`. "
)
# Retriever specific flags
FLAG_QUERY_END = flags.DEFINE_integer(
    "query_end",
    256,
    "When querying once, length of the query being taken from the inputs."
)
# FLAG_RETRIEVER_CONFIG_PATH = flags.DEFINE_string(
#     "retriever_config_path",
#     None,
#     "Path to the configuration file for the retrievers."
# )
FLAG_SCANN_CONFIG_PATH = flags.DEFINE_string(
    "scann_config_path",
    os.path.join(
        SCRIPT_DIRECTORY, "configs", "scann_configs", "default_config.json"
    ),
    "Configuration file for the ScaNN MIPS library."
)
FLAG_NUM_RETRIEVALS = flags.DEFINE_integer(
    "num_retrievals",
    None,
    "Number of neighbors to get with each retrieval."
    )
FLAG_RETRIEVAL_TEMPERATURE = flags.DEFINE_float(
    "retrieval_temperature",
    None,
    "Temperature to be used with the sampling in the softmax of certain "
    "retrievers (just retrievers.FullyCacherRetriever currently)."
)
FLAG_FULLYCACHED_H5_PATH = flags.DEFINE_string(
    "fullycached_h5_path",
    None,
    "Path to the .h5 file to be used by the fully cached retriever."
)
FLAG_RETRIEVAL_BANK_SIZE = flags.DEFINE_integer(
    "retrieval_bank_size",
    10,
    "Number of segments to sample from for the retrievals."
)

# Dataset specific flags
FLAG_DATASET_DEBUG = flags.DEFINE_boolean(
    "dataset_debug",
    False,
    "Whether to enable costly runtime checks for the dataset."
)
FLAG_INPUT_FIXED_SIZE = flags.DEFINE_boolean(
    "input_fixed_sized",
    True,
    "Whether to pad all inputs to the same size.")
FLAG_DATASET_NAME = flags.DEFINE_enum(
    "dataset_name",
    None,
    constants.DatasetNameChoices.choices(),
    "Name or TFDS key of the dataset we want"
)
FLAG_USE_SUBSET = flags.DEFINE_bool(
    "use_subset",
    False,
    "Whether to just use a subset of the data."
)
FLAG_SUBSET_SIZE = flags.DEFINE_integer(
    "subset_size",
    1000,
    "If we are using a subset of the data number of samples to use."
)

FLAG_DATASET_TYPE = flags.DEFINE_enum(
    "dataset_type",
    constants.DatasetTypeChoices.tfr,
    constants.DatasetTypeChoices.choices(),
    "Use TFR. Used to have more choices."
)
FLAG_QTY_SHUFFLE = flags.DEFINE_integer(
    "qty_shuffle",
    100,
    "Shuffle how many samples every time."
)
FLAG_TFR_PREFIX = flags.DEFINE_string(
    "tfr_prefix",
    None,
    "Prefix of the location of the tf record dataset.",
)
FLAG_MAX_LENGTH_GENERATION = flags.DEFINE_integer(
    "max_length_generation",
    None,
    "Maximum length of the generation."
)

FLAG_SAVE_PERIOD_MIN = flags.DEFINE_integer(
    "save-period-min",
    20,
    "How many minutes to wait between saves."
)

FLAG_TPU_NAME = flags.DEFINE_string(
  "tpu-name",
  socket.gethostname(),
  "Name of the TPU to use."
)

FLAG_OPTIMIZER_TYPE = flags.DEFINE_enum(
  "optimizer_type",
  None,
  constants.OptimizerTypes.choices(),
  "Which optimizer to use."
)

FLAG_LOG_SAMPLES = flags.DEFINE_boolean(
  "log_samples",
  None,
  "Whether to log the values of the samples. Very Costly!"
)
FLAG_DO_RESUME = flags.DEFINE_boolean(
  "do-resume",
  False,
  "Whether to resume training from a checkpoint."
)
FLAG_RESUME_PATH = flags.DEFINE_string(
  "resume-path",
  "",
  "From which path to resume from."
)
FLAG_TRAIN_ON_INPUT = flags.DEFINE_boolean(
  "train-on-input",
  False,
  "Whether to also train over the questions and the retrievals."
)

################################################################################
# Training and evaluation step functions.
################################################################################
# With tf.function, one can't pass non-tensor objects. This makes it so all
# non-tensor objects need to be passed through non-local references, making
# the step functions closures. In order to make our code cleaner / make
# dependencies more explicit, we build the closures with builder functions that
# explicitly show each step function's dependencies.
def build_regular_training_step(
    model,
    optimizer,
    strategy,
    tf_function_kwargs = None
):
  """Build the training step that is used in all cases but vertical mod. par."""
  tf_function_kwargs = {} if tf_function_kwargs is None else tf_function_kwargs

  @tf.function(**tf_function_kwargs)
  def training_step(input_ids, label_ids):
    """Computes the loss, backpropagates gradients, updates weights."""
    losses = []

    # According to the TF2 guide, there are advantages to doing multiple
    # batches in the same tf.function call
    with tf.GradientTape() as tape:
      partial_loss = model(
          input_ids,
          labels=label_ids,
          training=True,
          return_dict=True).loss
      if isinstance(partial_loss, values.PerReplica):
        average_loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, partial_loss, axis=None
        )
      else:
        average_loss = tf.math.reduce_mean(partial_loss)

      losses.append(average_loss)
      grads = tape.gfradient(average_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return tf.math.reduce_mean(losses)
  return training_step


# def build_manual_data_parallel_training_step(
#     models,
#     optimizer,
#     tf_function_kwargs = None
# ):
#   """Data parallel training step without using tf.distribute.Strategies."""
#
#   tf_function_kwargs = {} if tf_function_kwargs is None else tf_function_kwargs
#   train_vars = models[0].trainable_variables
#
#   @tf.function(**tf_function_kwargs)
#   def fn(input_ids, label_ids):
#     losses = []
#     accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]
#
#     # Get some gradients per model instance
#     for i in range(0, FLAG_BATCH_SIZE.value // FLAG_BATCH_SPLIT.value):
#       with tf.GradientTape() as tape:
#         start = i * FLAG_BATCH_SPLIT.value
#         end = (i + 1) * FLAG_BATCH_SPLIT.value
#         partial_loss = models[i](
#             input_ids[start:end],
#             labels=label_ids[start:end],
#             training=True,
#             return_dict=True
#         ).loss
#         gradients = tape.gradient(partial_loss, models[i].trainable_variables)
#         accum_gradient = [acum_grad + grad for acum_grad, grad
#                           in zip(accum_gradient, gradients)]
#       losses.append(partial_loss)
#
#     accum_gradient = [this_grad / len(models) for this_grad in accum_gradient]
#     # Apply to first model
#     optimizer.apply_gradients(
#         zip(accum_gradient, models[0].trainable_variables)
#         )
#
#     with tf.device("/job:localhost/replica:0/task:0/device:CPU:0"):
#       return tf.math.reduce_mean(losses)
#   return fn


def build_evaluation_step(
    model,
    tf_function_kwargs = None,
):
  # Can't assign {} to the default value, as assigning mutable values to
  # default value is a bad practice, warned against by the linter
  tf_function_kwargs = {} if tf_function_kwargs is None else tf_function_kwargs

  @tf.function(**tf_function_kwargs)
  def fn(input_ids, label_ids):
    losses = []
    for i in range(0, FLAG_BATCH_SIZE.value // FLAG_BATCH_SPLIT.value):
      start = i * FLAG_BATCH_SPLIT.value
      end = (i + 1) * FLAG_BATCH_SPLIT.value
      loss = model(
          input_ids[start:end],
          labels=label_ids[start:end],
          training=False,
          return_dict=True).loss
      losses.append(loss)

    return tf.math.reduce_mean(losses)
  return fn


def save_model(
    *,
    train_steps,
    model_or_replicas,
    optimizer,
    instance_output_dir
):
  """Save the model and log the flags, locally, then copy over to GS."""
  with tempfile.TemporaryDirectory() as tmp:
    save_directory = os.path.join(
        tmp,
        time.strftime(f"{train_steps}_ckpt_%Y%m%d-%H%M%S")
    )
    model_or_replicas.save_pretrained(os.path.join(save_directory, "model"))

    command = shlex.join(
        [
            "gsutil",
            "-m",
            "rsync",
            "-r",
            save_directory,
            instance_output_dir,
        ]
    )
    LOGGER.debug("Sending model. Command:\n\t- `%s`", command)
    subprocess.Popen(
      command,
      shell=True,
      # `shell=True` makes it so we let the shell find which gsutil to use.
      # `shlex.join` makes the join safe.
    )


def main(argv):
  ##############################################################################
  # Initial Setup. Logging, Flags, Random seeds.
  ##############################################################################
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  absl_logging.use_python_logging()
  flags_dict = {
      flag.name: flag.value
      for flag in FLAGS.flags_by_module_dict()[argv[0]]
  }

  if FLAGS.use_subset:
    message = (f"{colorama.Back.RED}{colorama.Fore.WHITE}"
               f"{colorama.Style.BRIGHT}USING A SUBSET OF THE DATASET"
               f"{colorama.Style.RESET_ALL}")
    LOGGER.warning(
        message
    )

  utils.log_module_args(LOGGER, argv[0])
  if not FLAGS.output_dir.startswith("gs://"):
    utils.check_exists(FLAG_OUTPUT_DIR.value)
    if not tf.io.gfile.isdir(FLAG_OUTPUT_DIR.value):
      raise RuntimeError("Output dir needs to be a directory.")

  tf.random.set_seed(FLAG_RANDOM_SEED.value)
  np.random.seed(FLAG_RANDOM_SEED.value)

  # Prepare the instance output directory path and save the config there
  # Prepare the path
  folder_name = time.strftime(
      f"{FLAG_RUN_NAME.value}_{FLAG_APPROACH_TYPE.value}_%Y%m%d-%H%M%S"
  )
  instance_output_dir = os.path.join(FLAG_OUTPUT_DIR.value, folder_name).strip()
  if not instance_output_dir.endswith("/"):
    instance_output_dir += "/"
  json_target = os.path.join(instance_output_dir, "training_params.json")

  # Make the folder if we're not on gcloud
  if not json_target.strip().startswith("gs://"):
    subprocess.check_call(["mkdir", "-p", instance_output_dir])

  # Safe the config file
  utils.to_json_file(json_target, flags_dict)

  ##############################################################################
  # Initialization and Configuration of the Devices.
  ##############################################################################
  tpu_setup = None
  # current_acelerator_type is always "CPU" in the beginning with TPUs
  if tf_utils.current_accelerator_type() == "CPU":
    tpu_setup = tf_utils.init_tpus(FLAG_TPU_NAME.value)

  LOGGER.debug("Devices we are computing on:\n%s",
               utils.wrap_iterable(map(str, tf_utils.devices_to_use())))
  LOGGER.debug("All devices:")
  LOGGER.debug(tf_utils.device_mapping())

  if tf_utils.current_accelerator_type() == "GPU":
    tf.config.set_soft_device_placement(True)

  if tf_utils.current_accelerator_type() != "TPU":
    tf.debugging.set_log_device_placement(True)

  utils.check_operator(
    operator.ne, tf_utils.current_accelerator_type(), "CPU"
  )

  assert FLAG_TPU_NAME.value == socket.gethostname(), (
    "This is a configuration choice. You can remove this. "
    "There will be no side effects.")

  if FLAG_DISTRIBUTE_MODE.value in constants.PURE_DATA_PARALLEL_STRATEGIES:
    actual_num_replicas = len(tf_utils.devices_to_use())
  elif FLAG_DISTRIBUTE_MODE.value in constants.DATA_PARALLEL_DMC:
    actual_num_replicas = FLAG_NUM_REPLICAS.value
  else:
    actual_num_replicas = 1

  ##############################################################################
  # We load the retriever model if it is needed.
  ##############################################################################
  # Not currently used. See old commits.
  retriever = None

  ##############################################################################
  # Distributed training task
  ##############################################################################
  if FLAG_TASK.value == constants.TaskChoices.train:
    with utils.log_duration(LOGGER, "main", "Load model"):
      utils.print_mem("before loading model", LOGGER)
      model_specific = task_specific.load_model(
          FLAG_MODEL_KEY.value,
          FLAG_DISTRIBUTE_MODE.value,
          tpu_setup,
          FLAG_NUM_REPLICAS.value
      )
      utils.print_mem("after loading model", LOGGER)
      model_or_replicas = model_specific.model
      if isinstance(model_or_replicas, list):
        model_or_replicas: List[transformers.TFGPT2LMHeadModel]
      else:
        model_or_replicas: transformers.TFGPT2LMHeadModel

      tokenizer = model_specific.tokenizer

      def make_optimizer():
        if FLAG_OPTIMIZER_TYPE.value == constants.OptimizerTypes.adafactor:
          return tensor2tensor.utils.adafactor.AdafactorOptimizer(
              learning_rate=FLAG_LEARNING_RATE.value
          )
        elif FLAG_OPTIMIZER_TYPE.value == constants.OptimizerTypes.adam:
          return tf.keras.optimizers.Adam(
            learning_rate=FLAG_LEARNING_RATE.value
          )
        else:
          raise ValueError(FLAG_OPTIMIZER_TYPE.value)

      if model_specific.strategy:
        with model_specific.strategy.scope():
          optimizer = make_optimizer()
      else:
        optimizer = make_optimizer()

    ############################################################################
    # Prepare the dataset functions
    ############################################################################
    rg = np.random.default_rng(FLAG_RANDOM_SEED.value)

    def call_lm_preproc(
        repeat,
        split,
        random_seed
    ):
      """Using functools.partial prevents the linter from doing its job."""
      if FLAG_DATASET_NAME.value == constants.DatasetNameChoices.kilt_eli5:
        return task_specific.create_lm_ds_kilt_eli5(
            tokenizer=tokenizer,
            context_window_size=model_or_replicas.config.n_positions,
            dataset_name=FLAG_DATASET_NAME.value,
            # Batches are split over the replicas:
            batch_size=FLAG_BATCH_SIZE.value * actual_num_replicas,
            db_path=FLAG_DB_PATH.value,
            random_seed=random_seed,
            use_subset=FLAG_USE_SUBSET.value,
            subset_size=FLAG_SUBSET_SIZE.value,
            use_helper_words=FLAG_USE_HELPER_WORDS.value,
            approach_type=FLAG_APPROACH_TYPE.value,
            num_retrievals=FLAG_NUM_RETRIEVALS.value,
            retrieval_temperature=FLAG_RETRIEVAL_TEMPERATURE.value,
            retriever=retriever,
            repeat=repeat,
            split=split,
            enable_debug_checks=FLAG_DATASET_DEBUG.value,
            retrieval_bank_size=FLAG_RETRIEVAL_BANK_SIZE.value,
            dataset_type=FLAG_DATASET_TYPE.value,
            qty_shuffle=FLAG_QTY_SHUFFLE.value,
            tfr_prefix=FLAG_TFR_PREFIX.value,
            max_length_generation=FLAG_MAX_LENGTH_GENERATION.value,
        )
      else:
        raise NotImplementedError(
            f"FLAG_DATASET_NAME.value unsupported: `{FLAG_DATASET_NAME.value}`"
        )

    make_training_dataset: Callable[..., tf.data.Dataset] = functools.partial(
        call_lm_preproc,
        split="train",
        repeat=False,
    )
    make_eval_dataset: Callable[..., tf.data.Dataset] = functools.partial(
        call_lm_preproc,
        split="eval",
        repeat=True,
    )

    ############################################################################
    # Prepare the step functions
    ############################################################################
    utils.check_contained(
        FLAG_DISTRIBUTE_MODE.value, constants.DistributeModeChoices.choices()
    )
    tf_function_flags = dict(
        experimental_compile=FLAG_EXPERIMENTAL_COMPILE.value,
        experimental_relax_shapes=not FLAG_INPUT_FIXED_SIZE.value
    )

    training_step = build_regular_training_step(
        model_or_replicas,
        optimizer,
        strategy=model_specific.strategy,
        tf_function_kwargs=tf_function_flags
    )

    evaluation_step = build_evaluation_step(
        model_or_replicas, tf_function_flags
    )

    secs_since_last_ckpt = time.time()
    # Model checkpoints are saved to the tmp_directory and then rsynced to GCS

    ############################################################################
    # Prepare the statistics and the logging facilities.
    ############################################################################
    # Tensorboard
    train_log_dir = os.path.join(instance_output_dir, "tensorboard", "train")
    eval_log_dir = os.path.join(instance_output_dir, "tensorboard", "eval")
    flags_log_dir = os.path.join(instance_output_dir, "tensorboard", "params")
    writers = dict(
        train=tf.summary.create_file_writer(train_log_dir),
        eval=tf.summary.create_file_writer(eval_log_dir),
        flags=tf.summary.create_file_writer(flags_log_dir)
    )
    with writers["flags"].as_default():
      tf.summary.text(
          "Flags",
          # Tensorboard takes Markdown:
          json.dumps(flags_dict, indent=4).replace("\n", "\n\n"),
          step=0
          )

    # Different information to log.
    ma_loss = dict(
        train=utils.MovingAverage(0.9),
        eval=utils.MovingAverage(0.9)
        )
    step_counters = dict(train=0, eval=0)
    batch_counters = dict(train=0, eval=0)
    prev_batch_end = time.time()

    ############################################################################
    # Create the Eval DS object.
    # ==========================================================================
    # The eval ds has no real concept of epoch, repeats forever, shuffling
    # each time it reaches its end.
    ############################################################################
    # Create
    with utils.log_duration(LOGGER, "main", "All of make_eval_dataset"):
      eval_ds_instance = make_eval_dataset(
          random_seed=rg.integers(-2**63, 2**63 - 1),
      )
    # Maybe distribute
    LOGGER.debug("Distributing the eval dataset to the replicas.")
    if FLAG_DATASET_TYPE.value == "tfr":
      eval_ds_instance = (
          model_specific.strategy.experimental_distribute_dataset(
              eval_ds_instance
          )
      )
    # Start the iteration. We step by calling `next(...)`.
    LOGGER.debug("Done distributing the eval dataset to the replicas.")
    eval_ds_instance = iter(eval_ds_instance)
    step_function = dict(train=training_step, eval=evaluation_step)

    ############################################################################
    # Training Loop
    # ==========================================================================
    # Create a new training dataset object that lasts for one epoch.
    # This is different from the eval training dataset object, which loops
    # forever.
    ############################################################################
    for epoch in itertools.count():
      ##########################################################################
      # Epoch Setup
      ##########################################################################
      LOGGER.debug("EPOCH %d START", epoch)
      # Shuffle differently every epoch
      with utils.log_duration(
          LOGGER, "main", "All of make_training_dataset"
      ):
        train_ds_instance = make_training_dataset(
            random_seed=rg.integers(-2**63, 2**63 - 1),
        )
      LOGGER.debug(
          "Attempting to distribute the training dataset to the replicas."
      )
      if FLAG_DATASET_TYPE.value == "tfr":
        train_ds_instance = (
            model_specific.strategy.experimental_distribute_dataset(
                train_ds_instance
            )
        )

      LOGGER.debug(
          "Done distributing the training dataset to the replicas."
      )
      train_ds_instance = iter(train_ds_instance)

      # To change splits, we use `itertools.islice` over the dataset generator.
      # When the training dataset generator is done, a new loop of the following
      # while loop occurs, but no training batch is done because we are taking
      # an `islice` of a generator that is done.
      did_at_least_one_training_batch = True
      split = "eval"
      while did_at_least_one_training_batch:
        utils.check_operator(
          operator.ne, tf_utils.current_accelerator_type(), "CPU"
        )

        # Invert split
        if split == "train":
          split = "eval"
        else:
          split = "train"

        # Prepare to test if we did at least one training batch
        if split == "train":
          did_at_least_one_training_batch = False

        ########################################################################
        # Take slices from the dataset iterator
        # ======================================================================
        # We only want to do a certain number of batches before switching splits
        # We do this by using an `itertools.islice` of the dataset iterators.
        ########################################################################
        if split == "train":
          dataset_iterator = toolz.take(
              FLAG_BATCHES_BETWEEN_EVALS.value, train_ds_instance
          )
        else:
          # The evaluation dataset generator is infinite, reshuffles everytime
          # it gets to its end.
          # Still, we take a fixed size slice form that infinite generator.
          dataset_iterator = toolz.take(
              FLAG_NUMBER_EVAL_BATCHES.value, eval_ds_instance
          )

        LOGGER.debug("Batching")
        for batch in dataset_iterator:
          if FLAG_LOG_SAMPLES.value:
            ####################################################################
            # Print elements of the dataset
            ####################################################################
            # Make ourselves resistant to values possibly being a PerReplica
            # object
            LOGGER.warning(
              f"%(red)sLOGGING SAMPLES. THIS IS VERY SLOW.%(reset)s",
              dict(
                red=colorama.Fore.RED,
                reset=colorama.Style.RESET_ALL,
              )
            )
            is_distributed = isinstance(
              batch["input_ids"], values.PerReplica
            )
            for in_batch_idx in range(FLAG_BATCH_SIZE.value):
              for replica_idx in (
                  range(actual_num_replicas) if is_distributed
                  else [0]
              ):
                if is_distributed:
                  sample = {k: batch[k].values[replica_idx] for k in batch}
                else:
                  sample = batch

                # input_sentence = tokenizer.decode(
                #   [x for x in sample["input_ids"][i] if x != tokenizer.eos_token_id]
                # )

                # LOGGER.debug(
                #   "%sInput [%d / %d]%s:\n\"%s\"",
                #   colorama.Fore.GREEN,
                #   replica_idx + 1,
                #   actual_num_replicas,
                #   colorama.Style.RESET_ALL,
                #   input_sentence,
                # )
                #
                # answer = tokenizer.decode(
                #   [(x if x != -100 else 0) for x in sample["label_ids"][i]]
                # )
                # LOGGER.debug(
                #   "%sLabel [%d / %d]%s:\n\"%s\"",
                #   colorama.Fore.GREEN,
                #   replica_idx + 1,
                #   actual_num_replicas,
                #   colorama.Style.RESET_ALL,
                #   answer,
                # )

                cons = console.Console()
                sentences = table.Table()
                sentences.add_column("BPE Index", justify="center")
                sentences.add_column("Inputs", justify="center")
                sentences.add_column("Labels", justify="center")
                for bpe_idx, (x, y) in enumerate(itertools.zip_longest(
                    sample["input_ids"][in_batch_idx].numpy(),
                    sample["label_ids"][in_batch_idx].numpy(),
                    fillvalue=None,
                )):
                  x_w = tokenizer.decode([x]) if x >= 0 else f"[ {x} ]"
                  y_w = tokenizer.decode([y]) if y >= 0 else f"[ {y} ]"
                  sentences.add_row(str(bpe_idx), x_w, y_w)

                cons.print(sentences)

          # We only care about training epochs as, obviously, we don't train
          # over eval samples; the number of  eval samples seen only
          # contributes to lowering the variance in the evaluation of when to
          # do early stopping.
          if split == "train":
            did_at_least_one_training_batch = True

          input_ids = batch["input_ids"]
          label_ids = batch["label_ids"]

          # Per split step counter
          step_counters[split] += FLAG_BATCH_SIZE.value * actual_num_replicas
          batch_counters[split] += 1

          ######################################################################
          # Model step function.
          ######################################################################
          step_function_kwargs = dict(
                input_ids=input_ids,
                label_ids=label_ids,
            )

          utils.print_mem(f"[{split}] - Mem before `strategy.run`", LOGGER)
          LOGGER.debug("[%s] - Calling `strategy.run`", split)
          loss = model_specific.strategy.run(
            step_function[split],
            kwargs=step_function_kwargs
          )
          LOGGER.debug("[%s] - Done `strategy.run`", split)
          utils.print_mem(f"[{split}] - Mem after `strategy.run`", LOGGER)

          ####################################################################
          # End of logging step code / Logging and saving the model.
          ####################################################################
          if (FLAG_DISTRIBUTE_MODE.value in
              constants.PURE_DATA_PARALLEL_STRATEGIES):
            utils.check_equal(len(loss.values), actual_num_replicas)
            LOGGER.debug(
              "[%s] - Real num replicas: %s",  split, actual_num_replicas
            )
            average_loss = float(tf.math.reduce_mean(loss.values).numpy())

            LOGGER.debug("[%s] - Loss: %s",  str(split), str(average_loss))

          else:
            average_loss = float(loss.numpy())


          tf.debugging.check_numerics(
            loss.values if isinstance(loss, values.PerReplica) else loss,
            "Numerics failed."
          )

          now = time.time()
          batch_duration = now - prev_batch_end
          prev_batch_end = now
          ma_loss[split].update(average_loss)

          LOGGER.info("[%s] - Epoch: # %d", split, epoch)
          LOGGER.info("[%s] - Tensorboard_dir: %s", split, instance_output_dir)
          LOGGER.info("[%s] - Batch: # %d", split, batch_counters[split])
          LOGGER.info("[%s] - Step:  # %d", split, step_counters[split])
          if FLAG_USE_SUBSET.value:
            LOGGER.warning(">> USING A SUBSET OF THE DATASET <<")
          LOGGER.info(
              "[%(split)s] - Batch loss:           %(metric)f",
              dict(split=split, metric=average_loss)
          )
          LOGGER.info(
              "[%(split)s] - Moving average loss:  %(metric)f",
              dict(split=split, metric=ma_loss[split].average)
          )
          LOGGER.info(
              "[%(split)s] - Moving average ppl:   %(metric)f",
              dict(split=split, metric=np.exp(ma_loss[split].average))
          )
          LOGGER.info(
              "[%(split)s] - Batch duration:       %(duration)s",
              dict(
                  split=split,
                  duration=utils.TimeStamp.from_seconds(
                      batch_duration).format()
              )
          )

          # Write to Tensorboard
          with writers[split].as_default():
            tf.summary.scalar(
                f"Loss/{split}", average_loss, step_counters[split]
            )
            tf.summary.scalar(
                f"PPL/{split}", np.exp(average_loss), step_counters[split]
            )
          writers[split].flush()


          ######################################################################
          # Save every `FLAG_SAVE_PERIOD_MIN.value` minutes.
          ######################################################################
          if (time.time() - secs_since_last_ckpt) / (
              60 * FLAG_SAVE_PERIOD_MIN.value) >= 1:
            dur = (time.time() - secs_since_last_ckpt) / (
                60 * FLAG_SAVE_PERIOD_MIN.value)
            secs_since_last_ckpt = time.time()
            LOGGER.debug("SAVING MODEL - CAUSE: DURATION - %0.2f min", dur)
            save_model(
                train_steps=step_counters["train"],
                model_or_replicas=model_or_replicas,
                optimizer=optimizer,
                instance_output_dir=instance_output_dir
            )

    #############################################################
    # Post Training Cleanup
    #######################################################################
    for writer in writers.values():
      writer.close()


if __name__ == "__main__":
  app.run(main)

