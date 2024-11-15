#!/bin/bash -e
BRANCH="dev"

# arguments for train_model.py
TRAIN_ARGS="--save_dir ./ReAct/outputs/ --dataset tinystories --group debug \
--num_blocks 4 --width 512 --n_heads 8 --epochs 1 --num_classes 50304 \
--log_interval 750 --save_interval 10000 --seqlen 512 \
--max_iters 3 --batch_size 256 --accum_steps 2 \
--strategy ddp --model_axis 1 \
--warmup_steps 180 --lr 2e-3 \
--beta_1 0.9 --beta_2 0.99 \
--weight_decay 3e-4 --drop_rate 0.00 --nesterov \
--exp_logging --tune_hyperparams --baseline"

# Export environment variables
export WANDB_API_KEY=78c7285b02548bf0c06dca38776c08bb6018593f
export HF_TOKEN=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
export HF_HOME=/workspace/junk/
export HF_DATASETS_CACHE=/workspace/junk/
export JAX_TRACEBACK_FILTERING=off

# Change to workspace directory
mkdir -p /workspace/junk/
cd /workspace/

# Git stuff
git clone -b $BRANCH https://github.com/neel04/ReAct_Jax.git
git config --global safe.directory '*'
cd ReAct_Jax/
git pull --all
cd ..

FLAG_FILE="/workspace/env_flag"

if [ ! -f "$FLAG_FILE" ]; then
    echo "Setting up environment..."
    # ------------------
    # Basics
    apt-get update
    apt-get install neovim tmux

    # Create virtual environment
    pip3 install uv
    uv venv 'main_env' --python 3.11
    source main_env/bin/activate

    # Install dependencies
    uv pip install -U "jax[cuda12]"
    uv pip install -q transformers datasets scalax tokenizers wandb einops tqdm jaxtyping optax optuna equinox rich
    uv pip install -U tensorflow tensorboard-plugin-profile optuna-integration lm-eval nvitop pdbpp
    uv pip install git+https://github.com/deepmind/jmp
    uv pip install git+https://github.com/Findus23/jax-array-info.git
    # ------------------
    # Create the flag file
    touch "$FLAG_FILE"
else
    echo "Reusing existing venv..."
fi

echo "Executing train_model.py"
source main_env/bin/activate
python3 ReAct_Jax/train_model.py $TRAIN_ARGS
echo "Finished training!"
