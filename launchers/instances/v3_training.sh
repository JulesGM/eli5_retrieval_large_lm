#!/usr/bin/env bash
set -e
source launchers/instances/base.sh

CONFIG=configs/launcher_configs/v3_training.json
run "$CONFIG" "$@"
