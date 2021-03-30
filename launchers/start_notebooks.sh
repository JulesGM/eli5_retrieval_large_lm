screen -S ngrok -dm ./ngrok http 8888
pushd eli5_retrieval_large_lm/notebooks
screen -S jupyter -dm jupyter lab
popd
