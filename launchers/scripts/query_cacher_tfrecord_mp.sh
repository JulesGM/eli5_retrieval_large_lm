#!/usr/bin/env bash

if [[ "$HOSTNAME" == MBP-* ]] ; then
  echo "Running on MBP, aborting"
  return
fi

SCRIPT_PATH=query_cacher_tfrecord_mp.py
CONFIG_PATH=configs/query_cacher_tfr_configs/remote.json

pytype "$SCRIPT_PATH" -P . --check-variable-types \
  --check-container-types \
  --check-parameter-types --precise-return && \
  python3 check_flags.py "$SCRIPT_PATH" && \
  FLAGS=$(python3 json_to_args.py "$CONFIG_PATH") && \
  python3 -u -m pdb -c continue "$SCRIPT_PATH" $FLAGS