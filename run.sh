#!/bin/bash
BRANCH="dev"
IMAGE_NAME="docker.io/neel04/react_image:latest"
CONTAINER_NAME="react_container"

# arguments for train_model.py
TRAIN_ARGS="--save_dir ./ReAct/outputs/ --dataset 'minipile' --group 'minipile' \
--num_blocks 4 --width 384 --n_heads 8 --max_iters 5 --epochs 2 --num_classes 50304 \
--log_interval 500 --save_interval 2000 --seqlen 512 \
--bf16 --accum_steps 2 --batch_size 512 \
--warmup_steps 500 --lr 4.5e-3 \
--weight_decay 5e-4 --drop_rate 0.01 \
--exp_logging"

# Stop all running Docker containers
echo "Stopping all running Docker containers..."

if ! timeout 300 sudo docker rm -f $CONTAINER_NAME; then
    echo "Command timed out. Restarting Docker daemon & retrying..."
    sudo systemctl restart docker
    sleep 10s; sudo docker rm -f $CONTAINER_NAME
fi

# Git stuff
git clone -b $BRANCH https://github.com/neel04/ReAct_Jax.git

sudo -s <<EOF
git config --global safe.directory '*'
cd ReAct_Jax/; git pull --all; cd ..

# Run the Docker container
echo "Running Docker container..."
docker run --pull 'always' -v $(pwd)/ReAct_Jax/:/ReAct_Jax/ -e EQX_ON_ERROR=nan -e PJRT_DEVICE=TPU -e XLA_USE_SPMD=1 --privileged --rm --net=host --name $CONTAINER_NAME -it -d $IMAGE_NAME

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