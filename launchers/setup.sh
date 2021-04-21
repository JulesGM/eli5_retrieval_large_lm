# Arguments:
# $1: commit id
# $2: Whether this is an alpha one-vm instance or not
# $3: ngrok token
# $4: INSTANCE_NAME
################################################################################
# Options
################################################################################
# set -x
set -e
set -u

################################################################################
# Definition of constants
################################################################################

RESETALL='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
LIGHTGRAY='\033[0;37m'
DARKGRAY='\033[1;30m'
LIGHTRED='\033[1;31m'
LIGHTGREEN='\033[1;32m'
YELLOW='\033[1;33m'
LIGHTBLUE='\033[1;34m'
LIGHTPURPLE='\033[1;35m'
LIGHTCYAN='\033[1;36m'
WHITE='\033[1;37m'

title () {
  local LINE="##################################################"
  echo -e "\n"
  echo -e "$LINE"
  echo -e "# $1"
  echo -e "$LINE"
}


################################################################################
# Parse Command Line Options
################################################################################
EXPECTED_NUM_ARGS=3
if [[ $# -lt $EXPECTED_NUM_ARGS ]]; then
    echo "Expected at least $EXPECTED_NUM_ARGS args. Got $#."
    exit 4
fi


GIT_COMMIT_ID="$1"
IS_ONE_VM_INSTANCE="$2"
NGROK_TOKEN="$3"
INSTANCE_NAME="$4"


if [[ "$IS_ONE_VM_INSTANCE" != "True" &&  "$IS_ONE_VM_INSTANCE" != "False" ]]
then
  echo "Expected \$2 to be either \"True\" or \"False\". Got \"$2.\""
  exit 4
fi


################################################################################
# Install Generic Dependencies
################################################################################
title "Installing generic dependencies"
echo -e "${ORANGE}Warning: apt-get takes a while to become available.${RESETALL}"
sudo apt-get -qq install -y wget 1>/dev/null


################################################################################
# Python, first part
################################################################################
if [[ "$IS_ONE_VM_INSTANCE" != "True" ]] ; then
  title "Downloading and installing Conda"
  # Download
  wget -q https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh \
      -O ~/Anaconda.sh 1>/dev/null

  # Run install file in automated mode
  bash ~/Anaconda.sh -b -p "$HOME/anaconda" 1>/dev/null
  ./anaconda/bin/conda init 1>/dev/null
  export PATH="/home/jules/anaconda/bin:$PATH"
  rm Anaconda.sh

  title "Updating Conda"
  conda upgrade -q --all -y 1>/dev/null
fi


title "Installing the basic Python dependencies"
python3 -m pip install cloud-tpu-client -q 1>/dev/null


################################################################################
# Git & Repo
################################################################################
title "Config the Git account"
git config --global user.email "jgagnonmarchand@gmail.com"
git config --global user.name "Jules Gagnon-Marchand"


if [[ ! -d eli5_retrieval_large_lm ]] ; then
  title "Download the project repo"
  git clone https://github.com/JulesGM/eli5_retrieval_large_lm.git \
    --recurse-submodules 1>/dev/null


  title "Checkout the correct commit and verify."
  pushd eli5_retrieval_large_lm
  echo "git checkout \"$GIT_COMMIT_ID\""
  git checkout "$GIT_COMMIT_ID"
  CHECKED_OUT_COMMIT_ID="$(git rev-parse HEAD)"
  popd
  if [[ "$GIT_COMMIT_ID" != "$CHECKED_OUT_COMMIT_ID" ]] ; then
    echo "Commit ids don't match:"
    echo -e "\tAs argument:   $1"
    echo -e "\tCurrent:       $1"
    exit
  fi
fi

################################################################################
# Rest
################################################################################
title "Installing all of the python requirements"
pushd eli5_retrieval_large_lm
if [[ "$IS_ONE_VM_INSTANCE" == "True" ]] ; then
  REQUIREMENTS_PATH=requirements_1vm_alpha.txt
else:
  REQUIREMENTS_PATH=requirements.txt
fi
python3 -m pip install -r "$REQUIREMENTS_PATH" -q 1>/dev/null
popd


title "Testing TPUs"
pushd eli5_retrieval_large_lm
python3 -c "
import sys
import tensorflow as tf
assert tf.__version__ == '2.5.0'
import tf_utils

if len(sys.argv):
  assert sys.argv[1] in {'True', 'False'}, sys.argv[1]

print(f'PYTHON TPU TEST ARGV: {sys.argv}')

if len(sys.argv) and sys.argv[1] == 'True':
  tf_utils.init_tpus(local=True)
elif len(sys.argv) > 1 and sys.argv[2]:
  tf_utils.init_tpus(tpu_name=sys.argv[1])
else:
  tf_utils.init_tpus()

print('\n'.join(map(str, tf_utils.devices_to_use())))
" "$IS_ONE_VM_INSTANCE" "$INSTANCE_NAME"
popd
exit

title "Installing gcsfuse"
sudo apt-get update
sudo apt-get install gcsfuse


title "Mounting the bucket"
LOG_PATH="$HOME/BUCKET-julesgm-research-v3"
mkdir "$LOG_PATH"
gcsfuse julesgm-research-v3 "$LOG_PATH"


title "Deleting old bashrc"
rm "$HOME/.bashrc"


title "Copying new bashrc"
cp "$HOME/bashrc" "$HOME/.bashrc"


title "Getting and setting up ngrok"
wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
unzip ngrok-stable-linux-amd64.zip
rm ngrok-stable-linux-amd64.zip
./ngrok authtoken "$NGROK_TOKEN"


title "Done. :)"