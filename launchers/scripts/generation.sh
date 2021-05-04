#!/usr/bin/env bash
set -e
set -u

#source venv/bin/activate

echo "Running base.sh"
source launchers/scripts/base.sh
echo "Done running bash.sh"

SCRIPT_PATH="generation.py"
CONFIG_PATH="configs/generation_configs/local_tpu.json"

LOG_DIR="$HOME/BUCKET-julesgm-research-v3/generation_run_log/"
LOG_PATH="$LOG_DIR/$(TZ=":America/New_York" date +"%Y-%m-%d_%T")"
mkdir -p "$LOG_DIR" || true

echo "Writing logs to '$LOG_PATH'."
run "$SCRIPT_PATH" "$CONFIG_PATH" "$@" 2>&1 | tee "$LOG_PATH"
