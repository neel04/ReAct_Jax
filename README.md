# ReAct_Jax
ReAct architecture and training loop - now in Jax!

### Docker

`run.sh` is the runner script for the Docker container. It pulls the latest version of the code from the `dev` branch, and runs `train_model.py` with the arguments specified in `TRAIN_ARGS`.

In the commands below, I download the script off a GitHub gist. You're better off forking this repo and changing the repo to directly `wget` from your fork.

The script takes care of actually pulling the `docker` container, the repo, and running the training loop across a TPU Pod slice automagically.

> **Make sure to modify the script with your own API tokens for WandB as well as fill the other flags at the top**

1. Run `run.sh` on your TPU pod slice (tested extensively with TPUv4-32)
2. Enjoy

## Inferencing

```bash
python3 inferencer.py --checkpoint_path '/Users/neel/Documents/research/ReAct_Jax/ReAct/outputs/model 5000.eqx' --num_blocks 3 --width 256 --n_heads 4 --seqlen 196  --prompt "Sam is sad because"
```

## Other commands

First, get a preemptible TPUv4-8 node as a queued resource:

```bash
gcloud alpha compute tpus queued-resources create node-v4 \
--node-id node-v4 \
--project react-jax \
--zone us-central2-b \
--accelerator-type v4-8 \
--runtime-version tpu-vm-v4-base \
--metadata-from-file startup-script=./run.sh \
--best-effort
```

Setup the TPU pod slice with basics:

```bash
gcloud compute tpus tpu-vm ssh node-v4 \
--zone=us-central2-b --worker=all --command="\
    sudo apt-get update; \
    sudo apt-get install neovim -y; \
    sudo snap install nvim --classic; \
    echo 'Setup done!"
```

And then actually kickoff the training by downloading the script and running it:

```bash
gcloud compute tpus tpu-vm ssh node-v4 \
--zone=us-central2-b --worker=all --command="\
    tmux kill-server; sudo rm -rf *; \
    wget https://gist.githubusercontent.com/neel04/3bfc7e4d9cd746829b7e72f1b6fac5de/raw/run.sh; \
    sleep 5s && tmux new-session -d 'bash run.sh'"
```