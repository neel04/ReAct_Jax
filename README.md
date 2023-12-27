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
TRAIN_ARGS="--save_dir ./ReAct/outputs/ --epochs 2 --warmup_steps 250 \
--lr 1.5e-3 --num_blocks 1 \
--width 192 --batch_size 256 --n_heads 2 --max_iters 5 \
--weight_decay 1e-4 --drop_rate 0.02  \
--log_interval 1000 --save_interval 1000 --seqlen 192  \
--bf16 --wandb"

git clone -b $BRANCH https://github.com/neel04/ReAct_Jax.git
git pull --all

# Stop all running Docker containers
echo "Stopping all running Docker containers..."
sudo docker stop $(sudo docker ps -a -q)

# Run the Docker container
echo "Running Docker container..."
sudo docker run --pull 'always' -v $(pwd)/ReAct_Jax/:/ReAct_Jax/ -e JAX_TRACEBACK_FILTERING=off --privileged --rm --net=host --name $CONTAINER_NAME -it -d $IMAGE_NAME

# Get docker container ID to copy files
CONTAINER_ID=$(sudo docker ps -aqf "name=$CONTAINER_NAME")
sudo docker cp $(pwd)/ReAct_Jax $CONTAINER_ID:/
export JAX_TRACEBACK_FILTERING=off

# Execute train_model.py inside the Docker container
echo "Executing train_model.py inside Docker container..."
sudo docker exec --privileged $CONTAINER_NAME python3 train_model.py $TRAIN_ARGS

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