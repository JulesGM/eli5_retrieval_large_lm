# Arguments:
# $1: commit id

# set -x
set -e
set -u
TPU_NAME=jules


if [[ -z "$1" ]] ; then
  echo "Didn't get a commit hash."
  exit
fi


title () {
  local LINE="##################################################"
  echo -e "\n"
  echo -e "$LINE"
  echo -e "# $1"
  echo -e "$LINE"
}


title "Installing generic dependencies"
sudo apt-get -qq install -y wget subversion 1>/dev/null


################################################################################
# Python, first part
################################################################################
title "Downloading and installing Conda"
# Download
wget -q https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh \
    -O ~/Anaconda.sh 1>/dev/null
# Run install file in automated mode
bash ~/Anaconda.sh -b -p "$HOME/anaconda" 1>/dev/null
./anaconda/bin/conda init 1>/dev/null
export PATH="/home/jules/anaconda/bin:$PATH"


title "Updating Conda"
conda upgrade -q --all -y 1>/dev/null


title "Installing Python dependencies"
if [[ "$(which python)" == "/usr/bin/python" ]]; then
  echo "Error: Wrong executable. Quitting."
  which python
  exit
fi
python -m pip install tf_nightly cloud-tpu-client -q 1>/dev/null


################################################################################
# Git & Repo
################################################################################
title "Config the Git account"
git config --global user.email "jgagnonmarchand@gmail.com"
git config --global user.name "Jules Gagnon-Marchand"


title "Download the project repo"
git clone https://github.com/JulesGM/eli5_retrieval_large_lm.git \
  --recurse-submodules 1>/dev/null


title "Checkout the correct commit and verify."
git checkout "$1"
CURRENT_COMMIT_ID="$(git rev-parse HEAD)"
if [[ "$1" != "$CURRENT_COMMIT_ID" ]] ; then
  echo "Commit ids don't match:"
  echo -e "\tAs argument:   $1"
  echo -e "\tCurrent:       $1"
  exit
fi


################################################################################
# Rest
################################################################################
title "Testing TPUs"
pushd eli5_retrieval_large_lm
python -c "
import sys
import tf_utils
tf_utils.init_tpus(sys.argv[1])
print('\n'.join(map(str, tf_utils.devices_to_use())))
" "$TPU_NAME"
popd


title "Installing all of the python requirements"
pushd eli5_retrieval_large_lm
python -m pip install -r requirements.txt -q 1>/dev/null
popd


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


title "Done. :)"