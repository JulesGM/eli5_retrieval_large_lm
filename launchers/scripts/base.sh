set -e

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
  IFS=" " read -a -r FLAGS <<< "$(python3 json_to_args.py "$CONFIG_PATH")"

  echo "FLAGS:"
  for FLAG in "${FLAGS[@]}" ; do
    echo " - \"$FLAG\""
  done

  pytype "$SCRIPT_PATH" -P . --check-variable-types \
    --check-container-types \
    --check-parameter-types --precise-return
  python3 check_flags.py "$SCRIPT_PATH"
  python3 -u -m pdb -c continue "$SCRIPT_PATH" "${FLAGS[@]}"
}