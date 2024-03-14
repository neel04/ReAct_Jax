# ReAct_Jax
ReAct architecture and training loop - now in Jax!

### Docker

This is the runner script for the Docker container. It pulls the latest version of the code from the `dev` branch, and runs `train_model.py` with the arguments specified in `TRAIN_ARGS`.

Thus you can easily modify the arguments in the below codeblock, and save the updated file somewhere. Everytime you run it, it would pull the latest git version on `BRANCH`.

> **Run below script with elevated permissions! `sudo`**

```bash
#!/bin/bash
BRANCH="dev"
IMAGE_NAME="docker.io/neel04/react_image:latest"
CONTAINER_NAME="react_container"

# arguments for train_model.py
TRAIN_ARGS="--save_dir ./ReAct/outputs/ --epochs 4 --warmup_steps 200 \
--lr 3e-4 --num_blocks 4 \
--width 384 --batch_size 512 --n_heads 4 --max_iters 5 \
--weight_decay 1e-3 --drop_rate 0.01 \
--log_interval 1000 --save_interval 1000 --seqlen 192 \
--bf16 --accum_steps 1 --exp_logging" #--tune_hyperparams"

# Stop all running Docker containers
echo "Stopping all running Docker containers..."
sudo docker stop $(sudo docker ps -a -q)

# Git stuff
git clone -b $BRANCH https://github.com/neel04/ReAct_Jax.git

sudo -s <<EOF
git config --global safe.directory '*'
cd ReAct_Jax/; git pull --all; cd ..

# Run the Docker container
echo "Running Docker container..."
docker run --pull 'always' -v $(pwd)/ReAct_Jax/:/ReAct_Jax/ -e EQX_ON_ERROR=nan --privileged --rm --net=host --name $CONTAINER_NAME -it -d $IMAGE_NAME

# Get docker container ID to copy files
CONTAINER_ID=$(docker ps -aqf "name=$CONTAINER_NAME")
docker cp $(pwd)/ReAct_Jax $CONTAINER_ID:/
export JAX_TRACEBACK_FILTERING=off

# Execute train_model.py inside the Docker container
echo "Executing train_model.py inside Docker container..."
docker exec --privileged $CONTAINER_NAME git config --global safe.directory '*'
docker exec --privileged $CONTAINER_NAME python3 train_model.py $TRAIN_ARGS
EOF

echo "Finished training!"
```

## Inferencing

```bash
python3 inferencer.py --checkpoint_path '/Users/neel/Documents/research/ReAct_Jax/ReAct/outputs/model 5000.eqx' --num_blocks 3 --width 256 --n_heads 4 --seqlen 196  --prompt "Sam is sad because"
```

## Other commands

Getting a preemptible TPUv4-8 node

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

Setup TPU pod slice - and ensure you have `run.sh` in the same directory as this command.

```bash
gcloud compute tpus tpu-vm scp run.sh node-v4: \
  --worker=all \
  --zone=us-central2-b

gcloud compute tpus tpu-vm ssh node-v4 \
--zone=us-central2-b --worker=all --command='\
    sudo apt-get update; \
    sudo apt-get install neovim -y; \
    sudo snap install nvim --classic; \
    wget https://gist.githubusercontent.com/neel04/3bfc7e4d9cd746829b7e72f1b6fac5de/raw/f1aa80c4a84affec84c9a70568764ea4f177091a/run.sh -O ./run.sh; \
    bash run.sh;'
```