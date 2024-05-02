# ReAct_Jax
ReAct architecture and training loop - now in Jax!

### Setup

`run.sh` is the runner script for the Docker container. It pulls the latest version of the code from the `dev` branch, and runs `train_model.py` with the arguments specified in `TRAIN_ARGS`.

In the commands below, I download the script off a GitHub gist. You're better off forking this repo and changing the repo to directly `wget` from your fork.

The script takes care of actually pulling the `docker` container, the repo, and running the training loop across a TPU Pod slice automagically.

> [!IMPORTANT]  
> **Make sure to modify & rebuild the Dockerfile with your own API tokens (for WandB & HuggingFace)**

> [!CAUTION]  
> Replace the docker image path (the envar `IMAGE_PATH`) in `run.sh` with your own DockerHub URL.

1. Run `run.sh` on your TPU pod slice (tested extensively with TPUv4-32)
2. Declare your `INSTANCE_NAME` from GCP via `export INSTANCE_NAME=<your_instance_name>`
3. Enjoy

## Inferencing

```bash
python3 inferencer.py --checkpoint_path '/Users/neel/Documents/research/ReAct_Jax/ReAct/outputs/model 5000.eqx' --num_blocks 3 --width 256 --n_heads 4 --seqlen 196  --prompt "Sam is sad because"
```

## Doing a training run

1. First, get a preemptible TPUv4 node as a queued resource:

```bash
gcloud alpha compute tpus queued-resources create $INSTANCE_NAME \
--node-id react-node \
--project react-jax \
--zone us-central2-b \
--accelerator-type v4-32 \
--runtime-version tpu-vm-v4-base \
--best-effort
```

2. (Optiona) Setup the TPU pod slice with basics (`nvim` + `NvChad`):

```bash
gcloud compute tpus tpu-vm ssh $INSTANCE_NAME \
--zone=us-central2-b --worker=all --command="\
    sudo apt-get update; \
    sudo snap install nvim --classic; \
    git clone https://github.com/NvChad/starter ~/.config/nvim && nvim; \
    echo 'Setup done!'"
```

3. And then actually kickoff the training by downloading the `run.sh` script and running it:

```bash
gcloud compute tpus tpu-vm ssh $INSTANCE_NAME \
--zone=us-central2-b --worker=all --command="\
    tmux kill-server; sudo rm -rf ./* \
    sleep 3s && wget https://gist.githubusercontent.com/neel04/3bfc7e4d9cd746829b7e72f1b6fac5de/raw/run.sh; \
    sleep 5s && tmux new-session -d 'bash run.sh &> out.log'"
```

## Troubleshooting

If you get errors regarding workers not being able to sync up at the distributed barrier, do:

```bash
gcloud compute tpus tpu-vm ssh --zone "us-central2-b" $INSTANCE_NAME --worker 'all' --project "react-jax" --command 'sudo docker system prune -f && sudo rm -rf ~/.cache;'
```

If Docker is unresponsive, just restart docker service:

```bash
gcloud compute tpus tpu-vm ssh --zone "us-central2-b" $INSTANCE_NAME --worker 'all' --project "react-jax" --command 'sudo systemctl restart docker'
```