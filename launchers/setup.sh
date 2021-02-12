# set -x
set -e
TPU_NAME=jules


title () {
    echo -e "\n"
    echo -e "##############################################################"
    echo -e "# $1"
    echo -e "##############################################################"
}


title "Installing generic dependencies"
sudo apt-get -qq install -y wget subversion 1>/dev/null


title "Downloading and installing Conda"
# Download
wget -q https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh \
    -O ~/Anaconda.sh 1>/dev/null
# Run install file in automated mode
bash ~/Anaconda.sh -b -p $HOME/anaconda 1>/dev/null
./anaconda/bin/conda init 1>/dev/null
export PATH="/home/jules/anaconda/bin:$PATH"


title "Updating Conda"
conda upgrade -q --all -y 1>/dev/null


title "Installing python dependencies"
if [[ "$(which python)" == "/usr/bin/python" ]]; then
  echo "Error: Wrong executable. Quitting."
  which python
  abort
fi
python -m pip install tf_nightly cloud-tpu-client \
  --use-feature=2020-resolver -q 1>/dev/null


title "Download the project repo"
git clone https://github.com/JulesGM/eli5_retrieval_large_lm.git \
    1>/dev/null


title "Testing TPUs"
pushd eli5_retrieval_large_lm
python -c "
import tf_utils as tu
tu.init_tpus('$TPU_NAME')
print('\n'.join(map(str, tu.devices_to_use())))
"
popd


title "Installing all of the python requirements"
pushd eli5_retrieval_large_lm
python -m pip install -r requirements.txt -q 1>/dev/null
popd