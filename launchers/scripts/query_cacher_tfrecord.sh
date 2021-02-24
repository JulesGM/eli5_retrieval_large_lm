#!/usr/bin/env bash
pytype query_cacher_tfrecord.py -P . --check-variable-types \
  --check-container-types \
  --check-parameter-types --precise-return && \
  python3 check_flags.py query_cacher_tfrecord.py && \
  FLAGS=$(python3 json_to_args.py configs/query_cacher_tfr_configs/remote.json) && \
  python3 -u -m pdb -c continue query_cacher_tfrecord.py $FLAGS
