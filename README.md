# ReAct_Jax
ReAct architecture and training loop - now in Jax!

```shell
export WANDB_API_KEY=...
export jax_threefry_partitionable=1

# Package installation
pip3 install -r requirements.txt

# Training
clear; python3 train_model.py \--save_dir /Users/neel/Documents/research/ReAct_Jax/ReAct/outputs/ \--lr 3e-5 --num_blocks 1 --epochs 175 --warmup_steps 100 \                                                                     
--cl_seqlen 10 --log_interval 200 \
--width 256 --batch_size 384 \
--weight_decay 3e-3 --drop_rate 0.2 \
--resume "neel/ReAct_Jax/bmc9v0f4 + 0"

# Inference
python3 inferencer.py --checkpoint_path './ReAct/outputs/model_200.eqx' --prompt '1010101010101
```