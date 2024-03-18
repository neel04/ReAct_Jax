a#!/bin/bash
BRANCH="dev"
IMAGE_NAME="docker.io/neel04/react_image:latest"
CONTAINER_NAME="react_container"

# arguments for train_model.py
TRAIN_ARGS="--save_dir ./ReAct/outputs/ --epochs 4 --warmup_steps 300 \
--lr 5e-3 --num_blocks 4 \
--width 384 --batch_size 512 --n_heads 4 --max_iters 5 \
--weight_decay 3e-4 --drop_rate 0.0001 \
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