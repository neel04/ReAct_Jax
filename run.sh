#!/bin/bash -e
BRANCH="dev"
IMAGE_NAME="docker.io/neel04/react_image:latest"
CONTAINER_NAME="react_container"
RAMDISK_PATH=$(pwd)/ReAct_Jax/ramdisk/
HF_TOKEN="..."
WANDB_TOKEN="..."

# arguments for train_model.py
TRAIN_ARGS="--save_dir ./ReAct/outputs/ --dataset 'minipile' --group 'minipile' \
--num_blocks 16 --width 1024 --n_heads 8 --epochs 10 --num_classes 50304 \
--log_interval 250 --save_interval 1500 --seqlen 512 \
--max_iters 3 --accum_steps 1 --batch_size 768 \
--warmup_steps 30 --lr 6e-1 \
--beta_1 0.9 --beta_2 0.99 \
--weight_decay 9e-4 --drop_rate 0.0 --grad_clip 0.9 \
--exp_logging"

# Stop all running Docker containers
echo "Stopping all running Docker containers..."
sudo docker stop $CONTAINER_NAME
sudo docker rm $CONTAINER_NAME

if ! sudo sh -c "timeout 150s docker rm -f $CONTAINER_NAME"; then
    echo "Command timed out. Restarting Docker daemon & retrying..."
    sudo systemctl restart docker
    sleep 10s; sudo docker rm -f $CONTAINER_NAME
fi

# Git stuff
git clone -b $BRANCH https://github.com/neel04/ReAct_Jax.git

sudo -s <<EOF
# Setup ramdisk
mkdir -p $RAMDISK_PATH
sudo mount -o size=200G -t tmpfs none $RAMDISK_PATH

git config --global safe.directory '*'
cd ReAct_Jax/; git pull --all; cd ..

# Run the Docker container
echo "Running Docker container..."
docker run --pull 'always' -v $(pwd)/ReAct_Jax/:/ReAct_Jax/ --mount type=tmpfs,destination=$RAMDISK_PATH,tmpfs-mode=1770,tmpfs-size=214748364800 -e HF_HOME=$RAMDISK_PATH -e HF_DATASETS_CACHE=$RAMDISK_PATH -e WANDB_API_KEY=$WANDB_TOKEN -e HF_TOKEN=$HF_TOKEN -e JAX_TRACEBACK_FILTERING=off --privileged --rm --net=host --name $CONTAINER_NAME -it -d $IMAGE_NAME

# Get docker container ID to copy files
CONTAINER_ID=$(docker ps -aqf "name=$CONTAINER_NAME")
docker cp $(pwd)/ReAct_Jax $CONTAINER_ID:/
export JAX_TRACEBACK_FILTERING=off

# Execute train_model.py inside the Docker container
echo "Executing train_model.py inside Docker container..."
docker exec --privileged $CONTAINER_NAME git config --global safe.directory '*'
docker exec --privileged $CONTAINER_NAME python3 train_model.py $TRAIN_ARGS

echo "Attempting graceful shutdown of Docker container..."
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME

# If graceful shutdown fails, force removal
if [ "$(docker inspect -f '{{.State.Running}}' $CONTAINER_NAME)" == "true" ]; then
    echo "Forcefully removing Docker container..."
    docker rm -f $CONTAINER_NAME
fi

EOF

echo "Finished training!"