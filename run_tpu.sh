#!/bin/bash -e
BRANCH="dev"

# Define path for RAM disk
export DISK_PATH="$HOME/workspace"
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache"

# Export environment variables pointing to the ramdisk
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

# Prepopulate RAM disk with bucket contents
echo "Copying contents from gs://hf-data-bucket to $DISK_PATH..."
gsutil -m cp -r gs://hf-data-bucket/huggingface_datasets "$DISK_PATH/"
echo "Copy complete."

# Adjust ownership to the current user
sudo chown -R $(whoami):$(whoami) "$DISK_PATH"

# Other environment variables
export HF_DATASETS_IN_MEMORY_MAX_SIZE=10000000000 # 10GB
export jax_threefry_partitionable=1
export WANDB_API_KEY=78c7285b02548bf0c06dca38776c08bb6018593f
export HF_TOKEN=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
export JAX_TRACEBACK_FILTERING=off

# arguments for train_model.py
TRAIN_ARGS="--save_dir ./ReAct/outputs/ --dataset owt --group owt_repro --exp_logging \
--log_interval 1500 --save_interval 10000 --seqlen 512 --num_classes 50304 \
--num_blocks 13 --width 1024 --n_heads 8 --epochs 1 --max_iters 3 \
--batch_size 512 --accum_steps 1 --warmup_steps 1000 \
--lr 9e-4 --beta_1 0.9 --beta_2 0.98 --nesterov \
--weight_decay 3e-3 --drop_rate 0.00 \
--tune_hyperparams --sweep_metadata _Muon_accum --resume"

git clone -b $BRANCH https://github.com/neel04/ReAct_Jax.git

FLAG_FILE="./env_flag"

git config --global safe.directory '*'
cd ReAct_Jax/
git pull --all
cd ..

if [ ! -f "$FLAG_FILE" ]; then
    echo "Setting up environment..."
    sudo apt-get update -y
    sudo apt-get install neovim tmux -y
    
    # Set default python to python3
    sudo ln -sf /usr/bin/python3 /usr/bin/python

    # Create virtual environment
    pip3 install uv
    source ~/.profile
    uv venv 'main_env' --python 3.11
    source main_env/bin/activate

    uv pip install --no-cache-dir "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --prerelease allow
    uv pip install -q transformers datasets scalax tokenizers icecream wandb einops torch tqdm jaxtyping optuna equinox rich
    uv pip install -U optuna-integration plotly lm-eval pdbpp
    uv pip install git+https://github.com/google-deepmind/optax.git
    uv pip install git+https://github.com/deepmind/jmp
    uv pip install git+https://github.com/Findus23/jax-array-info.git
    uv pip install -q tensorflow tensorboard-plugin-profile etils importlib_resources "cloud-tpu-profiler>=2.3.0"

    # ------------------
    # Create the flag file
    touch "$FLAG_FILE"
else
    echo "Reusing existing venv..."
fi

echo "Executing train_model.py"
source main_env/bin/activate

echo "Executing train_model.py inside uv venv..."
cd ReAct_Jax/
python3 train_model.py $TRAIN_ARGS

echo "Finished training!"

sudo umount "$DISK_PATH"
rm -rf "$DISK_PATH"
