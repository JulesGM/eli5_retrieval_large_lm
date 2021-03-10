#!/usr/bin/env bash
set -e
set -u
source launchers/scripts/base.sh
SCRIPT_PATH="main.py"
CONFIG_PATH="configs/train_configs/tpu_gpt2_eli5_kilt.json"

LOG_DIR="$HOME/BUCKET-julesgm-research-v3/train_run_log/"
LOG_PATH="$LOG_DIR/$(date +"%Y-%m-%d_%H:%M:%S")"
mkdir -p "$LOG_PATH" || true

run "$SCRIPT_PATH" "$CONFIG_PATH" | tee "$LOG_PATH"