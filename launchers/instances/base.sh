set -e
LINE="$(python3 -c "import os; print('#' * int(os.popen('stty size', 'r').read().split()[1]))")"

function h1 () {
  echo "$LINE"
  echo "# $1"
  echo "$LINE"
}

function h2 () {
  echo "$1"
}

function run () {
  h1 "About to run the instance-launching script."

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

  h2 "Path to config file:"
  echo " - \`$CONFIG_PATH\`"
  echo ""

  h2 "Flags from config file:"
  for FLAG in "${FLAGS[@]}" ; do
    echo " - \"$FLAG\""
  done
  if [[ "${#FLAGS[@]}" == 0 ]] ; then
    echo " [No flags were provided]"
  fi
  echo ""

  h2 "Extra flags provided manually:"
  for FLAG in "${OTHER_FLAGS[@]}" ; do
    echo " - \"$FLAG\""
  done
  if [[ "${#OTHER_FLAGS[@]}" == 0 ]] ; then
    echo " [No extra flags were provided]"
  fi
  echo ""

  h1 "Launching."
  pytype launchers/launch-instance.py -P . --check-variable-types \
    --check-container-types \
    --check-parameter-types --precise-return
  python check_flags.py launchers/launch-instance.py
  python launchers/launch-instance.py "${FLAGS[@]}" "${OTHER_FLAGS[@]}"
}