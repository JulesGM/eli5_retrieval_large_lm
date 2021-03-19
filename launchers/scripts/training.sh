#!/usr/bin/env bash
set -e
set -u

echo "Running base.sh"
source launchers/scripts/base.sh
echo "Done running bash.sh"

SCRIPT_PATH="main.py"
CONFIG_PATH="configs/train_configs/tpu_gpt2_eli5_kilt.json"

LOG_DIR="$HOME/BUCKET-julesgm-research-v3/train_run_log/"
LOG_PATH="$LOG_DIR/$(TZ=":America/New_York" date +"%Y-%m-%d_%T")"
mkdir -p "$LOG_DIR" || true

echo "Writing logs to '$LOG_PATH'."
run "$SCRIPT_PATH" "$CONFIG_PATH" 2>&1 | tee "$LOG_PATH"
