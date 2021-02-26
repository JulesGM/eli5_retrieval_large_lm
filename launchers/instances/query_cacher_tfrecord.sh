#!/usr/bin/env bash
set -e
source launchers/instances/base.sh

CONFIG=configs/launcher_configs/query_cacher_tfrecord.json
run "$CONFIG" "$@"
