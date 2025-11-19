#!/bin/bash -e
BRANCH="dev"

# Define path for RAM disk
export DISK_PATH="$HOME/workspace"
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache"
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

# Export environment variables pointing to the ramdisk
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$DISK_PATH/huggingface"
export HF_DATASETS_CACHE="$DISK_PATH/huggingface_datasets"

# Ensure target directory exists
mkdir -p "$DISK_PATH"

# Allocate RAM disk (ensure machine has > 350GB RAM available)
echo "Allocating 350GB RAM disk at $DISK_PATH..."
sudo mount -t tmpfs -o size=350G tmpfs "$DISK_PATH"
echo "RAM disk mounted."

# Install gsutil if necessary (part of google-cloud-cli)
if ! command -v gsutil &> /dev/null; then
echo "gsutil not found, installing google-cloud-cli..."
sudo apt-get update
sudo apt-get install -y google-cloud-cli
fi

# Adjust ownership to the current user
sudo chown -R $(whoami):$(whoami) "$DISK_PATH"

# Other environment variables
export jax_threefry_partitionable=1
export WANDB_API_KEY=78c7285b02548bf0c06dca38776c08bb6018593f
export HF_TOKEN=$(echo "aGZfandzQmFOaU1lbmduQkJDQm5HeHhVYmlxWm1YQnF0Q2xTaA==" | base64 -d)
export JAX_TRACEBACK_FILTERING=off
export DISABLE_MULTIPROC=1
export LMEVAL_HASHMM=0 #TODO: Remove at some point
export WANDB_INIT_TIMEOUT=240

# arguments for train_model.py
TRAIN_ARGS="--save_dir ./ReAct/outputs/ --dataset fineweb --group fineweb_100B \
--log_interval 1500 --save_interval 10000 --seqlen 512 --num_classes 50304 \
--num_blocks 18 --width 1024 --n_heads 16 --epochs 1 --max_iters 3 \
--batch_size 128 --accum_steps 4 --warmup_steps 300 \
--lr 6e-4 --beta_1 0.65 --beta_2 0.8 --nesterov \
--weight_decay 6e-3 --drop_rate 0.00 --optimizer_type adamw \
--rank 64 --strategy megatron --model_axis 2"


git clone -b $BRANCH https://github.com/neel04/ReAct_Jax.git

git config --global safe.directory '*'
cd ReAct_Jax/
git pull --all
cd ..

echo "Setting up environment..."
sudo apt-get update -y
sudo apt-get install neovim tmux -y

# Set default python to python3
sudo ln -sf /usr/bin/python3 /usr/bin/python

# Install uv and sync project dependencies (including TPU extra)
pip3 install -q uv
source ~/.profile
pushd ReAct_Jax >/dev/null
uv sync --extra tpu --python 3.11
uv pip install git+https://github.com/neel04/lm-evaluation-harness.git@debug/mp # TODO: Remove my fork
popd >/dev/null

# ------------------
# Create the flag file
echo "Reusing existing venv..."

echo "Executing train_model.py inside uv venv..."
cd ReAct_Jax/
uv run train_model.py $TRAIN_ARGS

echo "Finished training!"
