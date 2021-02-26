#!/usr/bin/env bash
source launchers/scripts/base.sh
SCRIPT_PATH=query_cacher_tfrecord_mp.py
CONFIG_PATH=configs/query_cacher_tfr_configs/remote.json
run "$SCRIPT_PATH" "$CONFIG_PATH"