#!/usr/bin/env bash
set -e
source launchers/scripts/base.sh
SCRIPT_PATH="main.py"
CONFIG_PATH="configs/train_configs/tpu_gpt2_eli5_kilt.json"
run "$SCRIPT_PATH" "$CONFIG_PATH"