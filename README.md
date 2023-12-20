# ReAct_Jax
ReAct architecture and training loop - now in Jax!

### Docker

```bash
sudo docker run -it --rm --privileged -e WANDB_API_KEY=78c7285b02548bf0c06dca38776c08bb6018593f --entrypoint /bin/bash neel04/react_image:latest
git clone -b dev https://github.com/neel04/ReAct_Jax.git

export jax_threefry_partitionable=1
export WANDB_API_KEY=78c7285b02548bf0c06dca38776c08bb6018593f 
cd ./ReAct_Jax

# Training
python3 train_model.py \
--save_dir ./ReAct_Jax/ReAct/outputs/ --epochs 2 --warmup_steps 250 \
--lr 1.5e-3 --num_blocks 2 \
--width 96 --batch_size 256 --n_heads 2 --max_iters 1 \
--weight_decay 1e-4 --drop_rate 0.02  \
--log_interval 1000 --save_interval 1000 --seqlen 192  \
--bf16 --wandb
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
--best-effort
```