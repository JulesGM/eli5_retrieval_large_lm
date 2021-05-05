set -e
# LINE="$(python3 -c "import os; print('#' * int(os.popen('stty size', 'r').read().split()[1]))")"
LINE="##################################################"


function h1 () {
  echo "$LINE"
  echo "# $1"
  echo "$LINE"
}

function h2 () {
  echo "$1"
}

function run () {
  ##############################################################################
  # Checks and setup
  ##############################################################################

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
  local OTHER_FLAGS=( "${@:3}" )
  local FLAGS
  IFS=" " read -r -a FLAGS <<< "$(python3 json_to_args.py "$CONFIG_PATH")"


  ##############################################################################
  h1 "Script launcher for \`$SCRIPT_PATH\`"
  ##############################################################################

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  h2 "Flags from configuration file \`$CONFIG_PATH\`:"
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  for FLAG in "${FLAGS[@]}" ; do
    echo " - \"$FLAG\""
  done
  if [[ "${#FLAGS[@]}" == 0 ]]; then
    echo " [No flags were provided]"
  fi
  echo ""

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  h2 "Extra flags provided manually:"
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  for FLAG in "${OTHER_FLAGS[@]}" ; do
    echo " - \"$FLAG\""
  done
  if [[ "${#OTHER_FLAGS[@]}" == 0 ]] ; then
    echo " [No extra flags were provided]"
  fi
  echo ""


  ##############################################################################
  h1 "Running the script."
  ##############################################################################
  python3 -m pytype "$SCRIPT_PATH" -P . --check-variable-types \
    --check-container-types \
    --check-parameter-types --precise-return
  python3 check_flags.py "$SCRIPT_PATH"
  # python3 -um pdb -c continue "$SCRIPT_PATH" "${FLAGS[@]}"
  python3 "$SCRIPT_PATH" "${FLAGS[@]}" "${OTHER_FLAGS[@]}"
}