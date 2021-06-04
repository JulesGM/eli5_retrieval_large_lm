print("stdlib")
import itertools
import logging
import os
import sys

print("third party")
import numpy as np
import rich
import rich.console
import tensorflow as tf
import transformers
import tqdm

DIR = os.getcwd()
# Add project dir to PYTHONPATH
sys.path.append(os.path.dirname(DIR))

print("first party")
import constants
import generation
import task_specific
import utils

print("done")
LOGGER = logging.getLogger(__name__)

# Args
APPROACH_TYPE = constants.ApproachTypeChoices.cached_pretok
SPLIT = constants.SplitChoices.eval
BATCH_SIZE = 3
NUM_ENTRIES = 4
DATA_PATH = "../../data/cached_pretok"

assert os.path.exists(DATA_PATH)

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-xl")
ds = generation.prep_ds_for_generation(dict(
    tokenizer=tokenizer,
    context_window_size=1024,
    dataset_name="kilt_eli5",
    batch_size=BATCH_SIZE,  # >> We set our own batch size elsewhere
    db_path=None,  # None,
    random_seed=0,
    use_subset=False,
    subset_size=-1,
    use_helper_words=constants.HelperWordModeChoices.multiple,
    approach_type=APPROACH_TYPE,
    num_retrievals=5,  # Will never change
    retrieval_temperature=1.,
    retriever=None,  # Cached retrievals don't need a retriever
    repeat=False,  # Will never change
    split=SPLIT,
    enable_debug_checks=False,
    retrieval_bank_size=10,  # Will never change
    dataset_type=constants.DatasetTypeChoices.tfr,
    tfr_prefix=DATA_PATH,
    qty_shuffle=1,  # Will never change
    max_length_generation=350
  ), tokenizer, BATCH_SIZE, SPLIT)

num_entries_in_split = (
  task_specific.DATASET_CARDINALITIES["kilt_eli5"][SPLIT]
)
entries_counter = tqdm.tqdm(total=num_entries_in_split)
for batch_no, batch in enumerate(itertools.islice(ds, NUM_ENTRIES)):
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Display the inputs and outputs.
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  rich_console = rich.console.Console(color_system="256")
  print_sample = generation.make_print_sample()

  assert not np.all(batch[0] == batch[1]), batch[0] == batch[1]
  
  with utils.log_duration(
      LOGGER, "main", "all of tokenizer.decode for a batch."
  ):
    for i in range(batch.shape[0]):
      print(f"{batch.shape = }")
      utils.check_equal(len(batch.shape), 2)
      utils.check_equal(batch.shape[0], BATCH_SIZE)
      tokens = batch.numpy()[i]
      input_text = tokenizer.decode(tokens)
      print(f"Batch {batch_no}, Sample {i} / {BATCH_SIZE} of batch:")
      print(f"\tNum tokens: {len(tokens)}")
      print_sample(
        input_text, f"input batch_no {batch_no}", rich_console
      )