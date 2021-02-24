#!/usr/bin/env bash
if [[ "$HOSTNAME" != MBP-* ]] ; then
  echo "Not running on MBP, aborting"
  return
fi

pytype launchers/launch.py -P . --check-variable-types \
  --check-container-types \
  --check-parameter-types --precise-return && \
  python check_flags.py launchers/launch.py && \
  FLAGS="$(python json_to_args.py configs/launcher_configs/query_cacher_tfrecord.json)" && \
  python launchers/launch.py $FLAGS