#!/bin/bash -e
BRANCH="dev"

# Export environment variables
export jax_threefry_partitionable=1
export WANDB_API_KEY=78c7285b02548bf0c06dca38776c08bb6018593f
export HF_TOKEN=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
export HF_HOME=/tmp/
export HF_DATASETS_CACHE=/tmp/
export JAX_TRACEBACK_FILTERING=off

# arguments for train_model.py
TRAIN_ARGS="--save_dir ./ReAct/outputs/ --dataset 'tinystories' --group 'owt' \
--num_blocks 8 --width 128 --n_heads 16 --epochs 1 --num_classes 50304 \
--log_interval 750 --save_interval 10000 --seqlen 512 \
--max_iters 3 --batch_size 128 --accum_steps 4 \
--strategy 'megatron' --model_axis 2 \
--warmup_steps 180 --lr 3e-3 \
--beta_1 0.9 --beta_2 0.99 \
--weight_decay 5e-4 --drop_rate 0.00 --nesterov \
--exp_logging --tune_hyperparams"

git clone -b $BRANCH https://github.com/neel04/ReAct_Jax.git

FLAG_FILE="./env_flag"

git config --global safe.directory '*'
cd ReAct_Jax/; git pull --all; git checkout f55d7bcc7d5243c66015d0bcab0e38946ad1b9dc; cd ..

if [ ! -f "$FLAG_FILE" ]; then
    echo "Setting up environment..."
    apt-get update
    apt-get install neovim tmux

    # Create virtual environment
    pip3 install uv
    uv venv 'main_env' --python 3.11
    source main_env/bin/activate

    uv pip install -U -q jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    uv pip install -q transformers datasets scalax tokenizers icecream wandb einops torch tqdm jaxtyping optax optuna equinox rich
    uv pip install -U optuna-integration plotly lm-eval pdbpp
    uv pip install git+https://github.com/deepmind/jmp
    uv pip install git+https://github.com/Findus23/jax-array-info.git
    uv pip install -q tensorboard-plugin-profile tensorboard etils importlib_resources "cloud-tpu-profiler>=2.3.0"

    # ------------------
    # Create the flag file
    touch "$FLAG_FILE"
else
    echo "Reusing existing venv..."
fi

echo "Executing train_model.py"
source main_env/bin/activate

echo "Executing train_model.py inside uv venv..."
python3 train_model.py $TRAIN_ARGS

echo "Finished training!"
