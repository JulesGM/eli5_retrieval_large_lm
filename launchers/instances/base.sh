set -e

function run () {
  if [[ "$HOSTNAME" != MBP-* ]] ; then
    echo "Not running on MBP, aborting"
    return
  fi

  if [[ "$1" == "" ]] ; then
    echo "Got no positional arguments, aborting. "
  fi

  local CONFIG_PATH="$1"
  local OTHER_FLAGS=( "${@:2}" )
  local FLAGS
  IFS=" " read -r -a FLAGS <<< "$(python json_to_args.py "$CONFIG_PATH")"

  echo "FLAGS:"
  for FLAG in "${FLAGS[@]}" ; do
    echo " - \"$FLAG\""
  done

  echo "OTHER FLAGS:"
  for FLAG in "${OTHER_FLAGS[@]}" ; do
    echo " - \"$FLAG\""
  done

  pytype launchers/launch.py -P . --check-variable-types \
    --check-container-types \
    --check-parameter-types --precise-return
  python check_flags.py launchers/launch.py
  python launchers/launch.py "${FLAGS[@]}" "${OTHER_FLAGS[@]}"
}