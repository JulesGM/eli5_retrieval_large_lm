#!/usr/bin/env bash
set -e
source launchers/instances/base.sh

CONFIG=configs/launcher_configs/alpha_training.json
run "$CONFIG" "$@"
