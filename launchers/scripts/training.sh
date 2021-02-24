#!/usr/bin/env bash

if [[ "$HOSTNAME" == MBP-* ]] ; then
  echo "Running on MBP, aborting"
  return
fi

SCRIPT_PATH=main.py
CONFIG_PATH=configs/train_configs/tpu_gpt2_eli5_kilt.json

pytype "$SCRIPT_PATH" -P . --check-variable-types \
  --check-container-types \
  --check-parameter-types --precise-return && \
  python3 check_flags.py "$SCRIPT_PATH" && \
  FLAGS=$(python3 json_to_args.py "$CONFIG_PATH") && \
  python3 -u -m pdb -c continue "$SCRIPT_PATH" $FLAGS
