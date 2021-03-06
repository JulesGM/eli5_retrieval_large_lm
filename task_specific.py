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

"""Dataset and model specific code.
"""
import logging
import numpy as np
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from absl import flags
import constants
import dataclasses
import rich.console
import rich.panel
print = rich.console.Console(color_system="256").print

import tensorflow as tf
import tf_utils
import transformers
import utils

# tf.config.run_functions_eagerly(True)


FLAGS = flags.FLAGS
LOGGER = logging.getLogger(__name__)

TokenizerType = Union[transformers.PreTrainedTokenizer,
                      transformers.PreTrainedTokenizerFast]


################################################################################
# Model Specific
################################################################################
@dataclasses.dataclass
class CreateModelReturn:
  tokenizer: TokenizerType
  model: Union[transformers.PreTrainedModel, List[transformers.PreTrainedModel]]
  strategy: Optional[tf.distribute.Strategy]


def load_model(
    model_key,
    distribute_mode,
    tpu_setup,
    num_replicas,
    ):
  """Tries to load the model.

  Logs duration and memory use. Logs additional information if loading the model
  fails.

  Args:
    model_key: Key used to select the correct model loading function from
      the MODEL_FACTORIES dict.
    distribute_mode: A string describing how the model is distributed.
    tpu_setup: TPU configuration information.
    num_replicas: Number of data parallelism replicas.

  Returns:
    Returns an object containing the tokenizer, the model and the strategy.

  Raises:
    RuntimeError: If model_load_path points to nothing.
  """
  if distribute_mode not in constants.DistributeModeChoices.choices():
    raise ValueError(f"Unsupported distribute_mode: `{distribute_mode}`")

  if distribute_mode == constants.DistributeModeChoices.tpustrategy:
    if tpu_setup:
      strategy = tf.distribute.TPUStrategy(
          tpu_setup.resolver,
      )
    else:
      strategy = tf.distribute.TPUStrategy()
  elif distribute_mode == constants.DistributeModeChoices.onedevicestrategy:
    # Test mode with a single device, possibly a CPU.
    strategy = tf.distribute.OneDeviceStrategy(tf_utils.devices_to_use()[0])
  else:
    raise NotImplementedError(distribute_mode)

  with strategy.scope():
    config: CreateModelReturn = MODEL_FACTORIES[model_key](
        model_key,
        distribute_mode,
        None  # The replicas are created by the tf.distribute.Strategy obj
    )
    config.strategy = strategy
  return config


def _create_gpt2(
    model_name,
    distribute_mode,
    num_replicas  # pylint: disable=unused-argument
):
  """Loads the tokenizer and the model for the GPT2 extra large model."""

  ##############################################################################
  # Load the tokenizer
  ##############################################################################
  LOGGER.debug("Loading the weights: `%s`", model_name)
  tokenizer = transformers.GPT2TokenizerFast.from_pretrained(model_name)
  LOGGER.debug("Done loading the tokenizer.")
  LOGGER.debug("Loading the model weights.")

  with utils.log_duration(LOGGER, "main", "Loading the model."):
    model = transformers.TFGPT2LMHeadModel.from_pretrained(
        model_name,
        )

  logging.debug("Done loading the %s model.", model_name)
  return CreateModelReturn(
      tokenizer=tokenizer,
      model=model,
      strategy=None,
  )

def make_parse_fn(split: str, context_window_size: int) -> Callable:
  description: Dict[str, tf.io.FixedLenFeature] = {
      constants.CTH5Fields.distances:
          tf.io.FixedLenFeature((), tf.string),
      constants.CTH5Fields.gpt2_retrieved_ids:
          tf.io.FixedLenFeature((), tf.string),
      constants.CTH5Fields.gpt2_question_ids_inputs:
          tf.io.FixedLenFeature((), tf.string),
  }
  if split != constants.SplitChoices.test:
    description[
        constants.CTH5Fields.gpt2_answer_ids_inputs
    ] = tf.io.FixedLenFeature((), tf.string)

  feature_dtypes: Dict[str, tf.dtypes] = {
      constants.CTH5Fields.distances:
          tf.float32,
      constants.CTH5Fields.gpt2_retrieved_ids:
          tf.int32,
      constants.CTH5Fields.gpt2_question_ids_inputs:
          tf.int32,
  }
  if split != constants.SplitChoices.test:
    feature_dtypes[
        constants.CTH5Fields.gpt2_answer_ids_inputs
    ] = tf.int32

  feature_shape: Dict[str, Tuple[int, Ellipsis]] = {
      constants.CTH5Fields.distances:
          (10,),
      constants.CTH5Fields.gpt2_retrieved_ids:
          (10, context_window_size,),
      constants.CTH5Fields.gpt2_question_ids_inputs:
          (context_window_size,),
  }
  if split != constants.SplitChoices.test:
    feature_shape[constants.CTH5Fields.gpt2_answer_ids_inputs] = (
        context_window_size
    )

  # @tf.function
  def parse(sample):
    example = tf.io.parse_single_example(sample, description)
    output = {}
    for k, v in example.items():
      output[k] = tf.io.parse_tensor(v, out_type=feature_dtypes[k])
      output[k].set_shape(feature_shape[k])
    return output

  return parse

################################################################################
# Dataset Specific
################################################################################
_HELPER_TEXT = {
  "question": "Question:\n",
  "context": "\nContext:\n",
  "answer": "\nAnswer:\n"
}
def create_lm_ds_kilt_eli5(
    *,
    tokenizer,
    context_window_size,
    dataset_name,  # pylint: disable=unused-argument
    batch_size,
    split,
    db_path,  # pylint: disable=unused-argument
    random_seed,
    use_subset,  # pylint: disable=unused-argument
    subset_size,  # pylint: disable=unused-argument
    repeat,
    use_helper_words,
    approach_type,
    retriever,
    num_retrievals,
    retrieval_temperature,
    enable_debug_checks,
    retrieval_bank_size,  # pylint: disable=unused-argument
    dataset_type,
    qty_shuffle,
    tfr_prefix,
    max_length_generation,
):
  """Dataset preparation function for the Kilt version of the ELI5 dataset.

  This is for when the dataset is consumed by language models.

  Args:
    tokenizer: Tokenizer of the reader model.
    context_window_size: Size of the context of the reader model.
      Not used here.
    dataset_name: Exact name of the dataset. Some datasets share the same
      function, with small specific differences. Not used here.
    batch_size: Size of the batch for the reader model.
    prefetch_size: How many batches to prefetch.
    split: The train, evaluation or test split.
    dataset_paths_root: Root directory of the datasets. Not used here.
    random_seed: Seed used to shuffle the dataset. Should change at each epoch.
    use_subset: Whether to use a subset of the data
    subset_size: Size of the subset
    repeat: Whether to repeat the dataset
    use_helper_words: Whether to add helper words in the merged samples.
    approach_type: Type of overall solution we are using.
    retriever: Object that does the retrieval.
    num_retrievals: Number of retrievals to do.
    retrieval_temperature: For the retrieval methods that do sampling, what
      temperature to use.
  Returns:
    A tf.data.Dataset object that generates input_ids and label_ids for the
    generator model.
  Raises:
    RuntimeError: If we didn't find any files with the glob pattern.
    RuntimeError: If we are using a dataset type that is not supported.
  """

  maybe_retrieve_and_merge = _make_maybe_retrieve_and_merge_fn(
      tokenizer=tokenizer,
      context_size=context_window_size,
      retriever=retriever,
      temperature=retrieval_temperature,
      num_retrievals=num_retrievals,
      ds_split=split,
      approach_type=approach_type,  # FLAG_APPROACH_TYPE.value
      use_helper_words=use_helper_words,  # FLAG_USE_HELPER_WORDS
      enable_debug_checks=enable_debug_checks,
      max_length_generation=max_length_generation,
  )
  utils.check_equal(dataset_type, constants.DatasetTypeChoices.tfr)
  glob_pattern = os.path.join(tfr_prefix, f"{split}*")
  filenames = list(tf.io.gfile.glob(glob_pattern))
  if not filenames:
    raise RuntimeError(
        f"filnames is empty. Glob pattern was: {glob_pattern}"
    )

  parse = make_parse_fn(split, context_window_size)

  ds = tf.data.TFRecordDataset(
    filenames=filenames,
    num_parallel_reads=tf.data.experimental.AUTOTUNE,
  )

  ds = ds.map(
    parse,
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
    deterministic=False,
  )

  if repeat:
    ds = ds.repeat()

  utils.check_not_none(random_seed)
  utils.check_not_none(qty_shuffle)
  ds = ds.shuffle(qty_shuffle, seed=random_seed)

  ds = ds.batch(
      batch_size,
      drop_remainder=split != constants.SplitChoices.test,
  )

  # We can't use parallel calls here, the huggingface Rust fast tokenizer
  # breaks with multiple threads. It seems to still be worth it over their
  # slow one though, vs using parallel threads.

  ds = ds.map(maybe_retrieve_and_merge)
  # return map(maybe_retrieve_and_merge, ds)
  return ds

  # return ds.prefetch(tf.data.experimental.AUTOTUNE)


def _make_maybe_retrieve_and_merge_fn(
    *,
    tokenizer,
    context_size,
    ds_split,
    approach_type,  # FLAG_APPROACH_TYPE.value
    use_helper_words,  # FLAG_USE_HELPER_WORDS
    retriever,  # pylint: disable=unused-argument
    temperature,
    num_retrievals,
    enable_debug_checks,
    max_length_generation,
    tf_function_kwargs=None,
):
  """Build the `maybe_retrieve_and_merge` closure."""
  tf_function_kwargs = {} if tf_function_kwargs is None else tf_function_kwargs
  not_test_split = ds_split != constants.SplitChoices.test

  # @tf.function(**tf_function_kwargs)
  def maybe_retrieve_and_merge(
      batch,
  ):
    """Retrieve if needed, then finalize the prep. for model consumption."""

    batch_size = tf.shape(batch[
        constants.CTH5Fields.gpt2_question_ids_inputs
    ])[0]

    # Prepare the question ids inputs
    question_ids_inputs = batch[constants.CTH5Fields.gpt2_question_ids_inputs]
    question_ids_inputs = tf.RaggedTensor.from_tensor(
        question_ids_inputs,
        padding=constants.RAGGED_PADDING_ID
    )

    # Prepare the answer ids inputs
    answer_ids_inputs = None
    answer_ids_labels = None
    if not_test_split:
      answer_ids_inputs = batch[constants.CTH5Fields.gpt2_answer_ids_inputs]
      answer_ids_inputs = tf.RaggedTensor.from_tensor(
          answer_ids_inputs,
          padding=constants.RAGGED_PADDING_ID
      )
      answer_ids_labels = answer_ids_inputs

    ############################################################################
    # Prepare the helper words
    ############################################################################
    helper_word_token_ids = None
    if use_helper_words:
      helper_word_token_ids = {}
      for k in _HELPER_TEXT:
        ids = tf.constant(tokenizer.encode(_HELPER_TEXT[k]), dtype=tf.int32)
        ids = tf.repeat(tf.expand_dims(ids, 0), batch_size, axis=0)
        helper_word_token_ids[k] = ids

      question_ids_inputs = tf.concat(
          [helper_word_token_ids["question"], question_ids_inputs],
          axis=1
      )

    ##########################################################################
    # W/ Cached Retrievals
    ##########################################################################
    label_ids = None
    if approach_type == constants.ApproachTypeChoices.cached_pretok:
      bpe_indices_gpt2 = batch[constants.CTH5Fields.gpt2_retrieved_ids]
      bpe_indices_gpt2 = tf.RaggedTensor.from_tensor(
          bpe_indices_gpt2,
          ragged_rank=2,
          padding=constants.RAGGED_PADDING_ID
      )

      distances = batch[constants.CTH5Fields.distances]
      input_ids, label_ids = _prepare_samples_w_retrieval(
          split=ds_split,
          batch_size=batch_size,
          question_ids_inputs=question_ids_inputs,
          answer_ids_inputs=(
              answer_ids_inputs if not_test_split else None
          ),
          gpt2_tokenized_retrieved=bpe_indices_gpt2,
          num_retrievals_to_use=num_retrievals,
          temperature=temperature,
          context_size=context_size,
          enable_debug_checks=enable_debug_checks,
          distances=distances,
          max_generation_length=max_length_generation,
          helper_word_token_ids=helper_word_token_ids,
          use_helper_words=constants.HelperWordModeChoices.multiple,
      )

    elif approach_type == constants.ApproachTypeChoices.naked_lm:
      ##########################################################################
      # Without Retrievals
      ##########################################################################
      if use_helper_words:
        question_ids_inputs = tf.concat([
            question_ids_inputs,
            helper_word_token_ids["answer"],
        ], axis=1)

      question_ids_labels = tf.ones_like(
          question_ids_inputs
      ) * constants.PPL_MASK_ID

      if not_test_split:
        input_ids = tf.concat((question_ids_inputs, answer_ids_inputs),
                              axis=1)
        label_ids = tf.concat((question_ids_labels, answer_ids_labels),
                              axis=1)
      else:
        input_ids = question_ids_inputs
    else:
      raise RuntimeError("Unnsupported approach_type value"
                         f" {approach_type}")

    ############################################################################
    # Finalize the preparation
    ############################################################################
    # Convert to dense tensors
    input_ids = input_ids.to_tensor(tokenizer.eos_token_id)

    if not_test_split:
      final_eos = tf.RaggedTensor.from_tensor(
          tokenizer.eos_token_id * tf.ones([batch_size, 1], dtype=tf.int32)
      )
      label_ids = tf.concat([label_ids, final_eos], axis=1)
      label_ids = label_ids.to_tensor(constants.PPL_MASK_ID)

    # All samples need to have at least one token != -100 (PPL_MASK_ID)
    if enable_debug_checks and not_test_split:
      not_any_padding = tf.reduce_any(
          label_ids != constants.PPL_MASK_ID, axis=1
      )
      none_has_padding = tf.math.reduce_all(
          not_any_padding
      )
      qty_doesnt_have_padding = tf.reduce_sum(
          tf.cast(not_any_padding))

      check_no_padding = tf.Assert(
          none_has_padding,
          [qty_doesnt_have_padding]
      )
      with tf.control_dependencies([check_no_padding]):
        label_ids = tf.identity(label_ids)

    # Limit size
    input_ids = input_ids[:, :context_size]
    if not_test_split:
      label_ids = label_ids[:, :context_size]

    ############################################################################
    # Pad `input_ids` and `label_ids` to context_size
    ############################################################################
    # Prepare the ones
    pad_qty = tf.math.maximum(
        0, tf.constant(context_size) - tf.shape(input_ids)[1]
    )
    padding_ones = tf.ones(
        [batch_size, pad_qty],
        dtype=input_ids.dtype
    )
    # Pad the inputs
    input_padding = tokenizer.eos_token_id * padding_ones
    input_ids = tf.concat((input_ids, input_padding), axis=1)

    # Pad the labels labels
    if not_test_split:
      pad_qty = tf.math.maximum(
          0, tf.constant(context_size) - tf.shape(label_ids)[1]
      )
      padding_ones = tf.ones(
          [batch_size, pad_qty],
          dtype=input_ids.dtype
      )
      label_padding = -100 * padding_ones
      label_ids = tf.concat((label_ids, label_padding), axis=1)

    # Make checks
    if enable_debug_checks:
      control_dependencies = []
      control_dependencies.append(tf.Assert(
          tf.math.reduce_all(input_ids != -1),
          [input_ids],
          name="NoMinusOnesInputs"
      ))
      if not_test_split:
        control_dependencies.append(tf.Assert(
            tf.math.reduce_all(label_ids != -1),
            [label_ids],
            name="NoMinusOnesLabel"
        ))
        control_dependencies.append(tf.Assert(
            tf.logical_not(
                tf.math.reduce_any(
                    tf.math.reduce_all(label_ids != -100, axis=1)
                )
            ),
            [label_ids],
            name="NotAllMinusOneHundred"
        ))
      with tf.control_dependencies(control_dependencies):
        input_ids = tf.identity(input_ids)

    return dict(
        input_ids=input_ids,
        label_ids=label_ids if not_test_split else None
    )

  return maybe_retrieve_and_merge


# @tf.function
def _tokenize_and_concat_while_loop(
    all_retrieved_contexts: tf_utils.TFTensorType,
    selected_context_indices: tf_utils.TFTensorType,
    num_retrievals_to_use: tf_utils.TFTensorType,
    batch_size: tf_utils.TFTensorType,
    helper_word_mode: constants.HelperWordModeChoices,
    context_helper_word_tokens: tf_utils.TFTensorType,
):
  tf_utils.check_tf_tensor(all_retrieved_contexts)
  tf_utils.check_tf_tensor(selected_context_indices)

  """Tokenizes and puts together the retrievals, per batch unit."""
  def condition(
      loop_index: tf.Tensor,
      _,  # pylint: disable=unused-argument
  ):
    """While we have concatenated fewer contexts than `num_retrievals_to_use`
    """
    return tf.less(loop_index, num_retrievals_to_use)

  def body(
      loop_index,
      previously_concat_contexts: tf.RaggedTensor,
  ):

    # Take the retrieved contexts associated to the context index associated
    # to the current loop index
    context_to_concat: tf.RaggedTensor = tf.gather(
      all_retrieved_contexts,
      selected_context_indices[:, loop_index],
      batch_dims=1
    )
    # print("")
    # print(f"{previously_concat_contexts.row_lengths() = }")
    # print(f"{context_to_concat.row_lengths() = }")
    # print("")

    # Concatenate the tokens of the new context to the previously concatenated
    # contexts. Possibly add helper words.
    if helper_word_mode == constants.HelperWordModeChoices.once:
      previously_concat_contexts = tf.concat([
        previously_concat_contexts,
        context_to_concat
      ], axis=1)
    elif helper_word_mode == constants.HelperWordModeChoices.multiple:
      previously_concat_contexts = tf.concat([
        previously_concat_contexts,
        context_helper_word_tokens,
        context_to_concat
      ], axis=1)
    else:
      raise RuntimeError(f"Unsupported helper_word_mode: {helper_word_mode}")

    # Increment the counter.
    return loop_index + 1, previously_concat_contexts

  if batch_size is None:
    raise RuntimeError("batch_size is `None`. This should not happen.")

  return tf.while_loop(
      condition, body, [
        0, # loop index
        tf.RaggedTensor.from_tensor(
              tf.zeros(
                  shape=(batch_size, 0),
                  dtype=tf.int32
              ),
          ) # previously concatenated contexts
      ])[1]

def _print_info(
    concat_retrieved_: tf.RaggedTensor, title, tokenizer, helper_word_token_ids,
):
  panel_text = []
  panel_text += [f"{concat_retrieved_.shape = }"]
  panel_text += [f"{concat_retrieved_.row_lengths(axis=-1) = }"]
  for batch_idx in range(concat_retrieved_.shape[0]):
    whole_text = tokenizer.decode(concat_retrieved_[batch_idx])
    text_array = np.array(whole_text.split())
    helper_text = tokenizer.decode(helper_word_token_ids['context'][0]).strip()
    num_context_tokens = np.sum(text_array == helper_text)
    panel_text += [f"{num_context_tokens = }"]
  print(rich.panel.Panel("\n\n".join(panel_text), title=title))

# @tf.function
def _prepare_samples_w_retrieval(
    split,
    batch_size,
    question_ids_inputs: tf_utils.TFTensorType,
    answer_ids_inputs: tf_utils.TFTensorType,
    gpt2_tokenized_retrieved: tf_utils.TFTensorType,
    distances,
    num_retrievals_to_use,
    temperature,
    context_size,
    enable_debug_checks,
    use_helper_words,
    helper_word_token_ids,
    max_generation_length
):
  utils.check_contained(
    use_helper_words,
    constants.HelperWordModeChoices.choices()
  )

  """Prepares the samples that use retrieval.
  In regards to helper words, we only use them once. This could be changed.
  It would have many advantages.
  """
  assert (split == constants.SplitChoices.test) == (
      answer_ids_inputs is None
  ), (split == constants.SplitChoices.test, answer_ids_inputs)

  tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-xl")
  # panel_title = "Begining of _prepare_samples_w_retrieval"
  # panel_text = [f"{question_ids_inputs.shape = }"]
  # panel_text += [f"{question_ids_inputs.row_lengths(axis=-1) = }"]
  # panel_text += [f"{answer_ids_inputs.shape = }"]
  # panel_text += [f"{answer_ids_inputs.row_lengths(axis=-1) = }"]
  # panel_text += [f"{distances.shape = }"]
  # panel_text += [f"{gpt2_tokenized_retrieved.shape = }"]
  # panel_text += [f"{gpt2_tokenized_retrieved.row_lengths(axis=-1) = }"]
  # print(rich.panel.Panel("\n\n".join(panel_text), title=panel_title))
  is_not_test = split != constants.SplitChoices.test

  if not isinstance(question_ids_inputs, tf.RaggedTensor):
    question_ids_inputs = tf.RaggedTensor.from_tensor(
        question_ids_inputs,
        padding=constants.RAGGED_PADDING_ID
    )

  if enable_debug_checks:
    asserts = []
    asserts.append(
        tf.Assert(
            tf.math.reduce_all(
                question_ids_inputs != constants.RAGGED_PADDING_ID,
            ),
            [question_ids_inputs.to_tensor()]
        )
    )
    if is_not_test:
      asserts.append(
          tf.Assert(
              tf.math.reduce_all(
                  answer_ids_inputs != constants.RAGGED_PADDING_ID,
              ),
              [answer_ids_inputs.to_tensor()]
          )
      )
    with tf.control_dependencies(asserts):
      question_ids_inputs = tf.identity(question_ids_inputs)

  # These checks are at graph composition time, so OK
  utils.check_isinstance(question_ids_inputs, tf.RaggedTensor)

  if is_not_test:
    utils.check_isinstance(answer_ids_inputs, tf.RaggedTensor)

  ##############################################################################
  # Sample from the possible retrievals
  ##############################################################################
  # Choose the indices
  selected_context_indices = tf_utils.sample_without_replacement(
      distances / temperature, num_retrievals_to_use
  )

  # Concatenate the retrievals
  utils.check_isinstance(helper_word_token_ids, dict)
  utils.check_isinstance(
    helper_word_token_ids['context'],
    tuple([np.ndarray] + list(tf_utils.TfTensorTypeTuple))
  )

  concat_retrieved = _tokenize_and_concat_while_loop(
      gpt2_tokenized_retrieved,
      selected_context_indices=selected_context_indices,
      batch_size=batch_size,
      num_retrievals_to_use=num_retrievals_to_use,
      helper_word_mode=use_helper_words,
      context_helper_word_tokens=helper_word_token_ids['context'],
  )

  if use_helper_words == constants.HelperWordModeChoices.once:
    concat_retrieved = tf.concat([
        helper_word_token_ids["context"],
        concat_retrieved,
    ], axis=1)
  # _print_info(
  #   concat_retrieved,
  #   f"Num of 'context' helper words. Mode: {use_helper_words}",
  #   tokenizer,
  #   helper_word_token_ids
  # )

  # Cut the lengths down to max_lens_retrieval.
  # The eventual length of the ["question"] helper_tokens is included in
  # question_ids_inputs.
  if is_not_test:
    max_lens_retrieval = (
        context_size * tf.ones(
            shape=(batch_size,),
            dtype=tf.int64,
        )
        - (question_ids_inputs.row_lengths() +
           # We always generate the same length of text.
           max_generation_length +  # answer_ids_inputs.row_lengths() +
           (helper_word_token_ids["answer"].shape[1] if use_helper_words else 0)
           )
    )

  else:
    max_lens_retrieval = (
        context_size * tf.ones(
            shape=(batch_size,),
            dtype=tf.int64,
        ) - (question_ids_inputs.row_lengths()  +
             max_generation_length +
             (helper_word_token_ids["answer"].shape[1]
              if use_helper_words else 0
              )
             )
    )

  concat_retrieved = tf.ragged.boolean_mask(
      concat_retrieved,
      (
          tf.ragged.range(concat_retrieved.row_lengths()) <
          tf.expand_dims(max_lens_retrieval, axis=1)
      )
  )
  panel_text = []
  panel_text += [f"{selected_context_indices.shape = }"]
  panel_text += [f"{concat_retrieved.shape = }"]
  panel_text += [f"{concat_retrieved.row_lengths(axis=-1) = }"]
  panel_text += [f"{max_lens_retrieval = }"]
  print(rich.panel.Panel("\n\n".join(panel_text)))

  if enable_debug_checks:
    asserts = [
        tf.Assert(
            tf.math.reduce_all(max_lens_retrieval < context_size),
            [max_lens_retrieval, context_size]
        ),
    ]
    with tf.control_dependencies(asserts):
      concat_retrieved = tf.identity(concat_retrieved)

  if use_helper_words:
    if is_not_test:
      new_input_ids = tf.concat(
          [question_ids_inputs,
           concat_retrieved,
           helper_word_token_ids["answer"],
           answer_ids_inputs
           ],
          axis=1
      )
      new_label_ids = tf.concat(
          [-100 * tf.ones_like(question_ids_inputs),
           -100 * tf.ones_like(concat_retrieved),
           -100 * tf.ones_like(helper_word_token_ids["answer"]),
           answer_ids_inputs
           ],
          axis=1
      )
    else:
      new_input_ids = tf.concat(
          [question_ids_inputs,
           concat_retrieved,
           helper_word_token_ids["answer"],
           ],
          axis=1
      )
  else:
    if is_not_test:
      new_input_ids = tf.concat(
          [question_ids_inputs,
           concat_retrieved,
           answer_ids_inputs
           ],
          axis=1
      )
      new_label_ids = tf.concat(
          [-100 * tf.ones_like(question_ids_inputs),
           -100 * tf.ones_like(concat_retrieved),
           answer_ids_inputs
           ],
          axis=1
      )
    else:
      new_input_ids = tf.concat(
          [question_ids_inputs,
           concat_retrieved,
           ],
          axis=1
      )

  new_input_ids : tf.RaggedTensor
  return new_input_ids, new_label_ids if is_not_test else None


################################################################################
# Varia
################################################################################

DATASET_CARDINALITIES = {
    constants.DatasetNameChoices.kilt_eli5: {
        "train": 272637,
        "eval": 1507,
        "test": 600,
    }
}

# Pick the correct model creation function from the Hugging Face Model key.
MODEL_FACTORIES = {
    "gpt2": _create_gpt2,
    "gpt2-medium": _create_gpt2,
    "gpt2-large": _create_gpt2,
    "gpt2-xl": _create_gpt2,
    "distilgpt2": _create_gpt2,
}


