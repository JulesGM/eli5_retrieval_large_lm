set -e

LINE="$(python -c "import os; print('#' * os.get_terminal_size(0)[0])")"

function h1 () {
  echo "$LINE"
  echo "# $1"
  echo "$LINE"
}

function h2 () {
  echo "$1"
}

function run () {
  if [[ "$HOSTNAME" == MBP-* ]] ; then
    echo "Running on MBP, aborting."
    return
  fi

  if [[ "$2" == "" || "$1" == "" ]] ; then
    echo "Got fewer than 2 positional arguments, aborting."
    return
  fi

  local SCRIPT_PATH="$1"
  local CONFIG_PATH="$2"
  local FLAGS
  IFS=" " read -r -a FLAGS <<< "$(python3 json_to_args.py "$CONFIG_PATH")"

  h1 "Script launcher for \`$SCRIPT_PATH\`"
  echo ""

  h2 "Flags from configuration file \`$CONFIG_PATH\`:"
  for FLAG in "${FLAGS[@]}" ; do
    echo " - \"$FLAG\""
  done
  echo ""

  h1 "Running the script."

  pytype "$SCRIPT_PATH" -P . --check-variable-types \
    --check-container-types \
    --check-parameter-types --precise-return
  python3 check_flags.py "$SCRIPT_PATH"
  python3 -u -m pdb -c continue "$SCRIPT_PATH" "${FLAGS[@]}"
}