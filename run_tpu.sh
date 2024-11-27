#!/bin/bash -e
BRANCH="dev"
IMAGE_NAME="docker.io/neel04/react_image:latest"
CONTAINER_NAME="react_container"
RAMDISK_PATH=$(pwd)/ReAct_Jax/ramdisk/

# arguments for train_model.py
TRAIN_ARGS="--save_dir ./ReAct/outputs/ --dataset 'owt' --group 'owt' \
--num_blocks 8 --width 128 --n_heads 16 --epochs 1 --num_classes 50304 \
--log_interval 750 --save_interval 10000 --seqlen 512 \
--max_iters 3 --batch_size 128 --accum_steps 4 \
--strategy 'megatron' --model_axis 2 \
--warmup_steps 180 --lr 3e-3 \
--beta_1 0.9 --beta_2 0.99 \
--weight_decay 5e-4 --drop_rate 0.00 --nesterov \
--exp_logging --tune_hyperparams"

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
git config --global safe.directory '*'
cd ReAct_Jax/; git pull --all; git checkout f55d7bcc7d5243c66015d0bcab0e38946ad1b9dc; cd ..

# Run the Docker container
echo "Running Docker container..."
docker run --ulimit core=0 --pull 'always' -v $(pwd)/ReAct_Jax/:/ReAct_Jax/ --mount type=tmpfs,destination=$RAMDISK_PATH,tmpfs-size=375000000000 -e HF_HOME=$RAMDISK_PATH -e HF_DATASETS_CACHE=$RAMDISK_PATH -e WANDB_API_KEY=78c7285b02548bf0c06dca38776c08bb6018593f -e HF_TOKEN=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 -e JAX_TRACEBACK_FILTERING=off -e LIBTPU_INIT_ARGS="--xla_tpu_impure_oom_fast_exit_threshold=-1" -e XLA_FLAGS="--xla_dump_to=dump/hlo-dump" --privileged --rm --net=host --name $CONTAINER_NAME -it -d $IMAGE_NAME

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
