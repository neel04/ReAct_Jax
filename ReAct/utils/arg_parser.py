import argparse
from argparse import Namespace


def parse_args() -> Namespace:
    description = "Training script - kicks off training of the model"
    epilog = "Hyperparameters defaults may not be optimal. Please use --help to check available options."

    parser = argparse.ArgumentParser(description=description, epilog=epilog)

    # Model
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=1,
        help="Number of attention blocks. Default: 1",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        help="Parallelization strategy to use. Check `sharding.py` for more info",
    )

    parser.add_argument(
        "--model_axis", type=int, default=1, help="Axis to shard the model on."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="openwebtext",
        help="Dataset to use. Default: openwebtext",
    )

    parser.add_argument(
        "--width", type=int, default=384, help="Width dimension. Default: 384"
    )

    parser.add_argument(
        "--rank", type=int, default=64, help="Rank dimension for LoRA. Default: 64"
    )

    parser.add_argument(
        "--drop_rate", type=float, default=0.1, help="Dropout rate. Default: 0.1"
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=50304,
        help="Number of target classes OR vocab size. Default: 50304",
    )

    parser.add_argument(
        "--n_heads", type=int, default=4, help="Number of attention heads. Default: 4"
    )

    parser.add_argument(
        "--max_iters", type=int, default=10, help="Number of iterations. Default: 10"
    )

    parser.add_argument(
        "--seqlen", type=int, default=512, help="Sequence length. Default: 512"
    )

    parser.add_argument(
        "--batch_size", type=int, default=208, help="Batch size. Default: 208"
    )

    parser.add_argument(
        "--lr", type=float, default=3e-5, help="Learning rate. Default: 3e-5"
    )

    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay. Default: 1e-5"
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps. Default: 100",
    )

    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping. Default: 1.0"
    )

    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adamw",
        help="Optimizer to use. Default: AdamW",
    )

    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs. Default: 1"
    )

    parser.add_argument(
        "--beta_1", type=float, default=0.9, help="Adam beta 1. Default: 0.9"
    )

    parser.add_argument(
        "--beta_2", type=float, default=0.999, help="Adam beta 2. Default: 0.999"
    )

    parser.add_argument(
        "--nesterov",
        action="store_true",
        default=False,
        help="Enable Nesterov momentum. Default: False",
    )

    parser.add_argument(
        "--save_interval", type=int, default=2500, help="Save interval. Default: 2500"
    )

    parser.add_argument(
        "--log_interval", type=int, default=1000, help="Log interval. Default: 1000"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="/Users/neel/Documents/research/ReAct_Jax/ReAct/outputs/",
        help="Save directory for checkpoints. Default: ./outputs/. Keep the slash at the end",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode - disables JIT. Default: False",
    )

    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Switch model weights to bf16. Grads are always in bf16. Default: False (fp32)",
    )

    parser.add_argument(
        "--exp_logging",
        action="store_true",
        default=False,
        help="Enable experiment logging in the cloud. Default: False",
    )

    parser.add_argument(
        "--group",
        type=str,
        default="OpenWebText",
        help="WandB group name. Default: OWT",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint. Default: None",
    )

    parser.add_argument(
        "--resume",
        nargs="?",
        default=False,
        const=True,  # If no string is provided after the flag
        help="Obtain WandB run_path from Overview tab and append the"
        'epoch & step number with a +. \nExample arg: "neel/ReAct_Jax/6ktmhalt/ + 0 + 200"',
    )

    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Train baseline vanilla transformer model. Default: False",
    )

    parser.add_argument(
        "--tune_hyperparams",
        action="store_true",
        default=False,
        help="Tune hyperparameters using wandb sweep. Default: False",
    )

    parser.add_argument(
        "--sweep_metadata",
        type=str,
        default="",
        help="Extra metadata for W&B Sweep Name. Always Prefix a `_`. Default: ''",
    )

    parser.add_argument(
        "--accum_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients over. Default: 1",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Enable TensorBoard profiling. Default: False",
    )

    args = parser.parse_args()
    return args


def get_inference_args() -> Namespace:
    description = "Inference script for sampling from a trained model"

    parser = argparse.ArgumentParser(description=description)

    # These defaults should be mostly fine

    parser.add_argument(
        "--seqlen", type=int, default=512, help="Sequence length. Default: 512"
    )

    parser.add_argument(
        "--max_iters", type=int, default=3, help="Number of iterations. Default: 3"
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=50304,
        help="Number of target classes OR vocab size. Default: 50304",
    )

    parser.add_argument(
        "--num_tokens",
        type=int,
        default=64,
        help="Number of output tokens to generate. Default: 64",
    )

    # Likely need to be overwritten

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./",
        help="Path for the model checkpoint. Default: ./",
    )

    parser.add_argument(
        "--n_heads", type=int, default=4, help="Number of attention heads. Default: 4"
    )

    parser.add_argument(
        "--width", type=int, default=384, help="Width dimension. Default: 384"
    )

    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Train baseline vanilla transformer model. Default: False",
    )

    parser.add_argument(
        "--num_blocks",
        type=int,
        default=1,
        help="Number of attention blocks. Default: 1",
    )

    parser.add_argument(
        "--system_prompt",
        type=str,
        default="",
        help="The user system prompt provided to the model",
    )

    parser.add_argument(
        "--prompt", type=str, default="", help="The user prompt provided to the model"
    )

    parser.add_argument(
        "--temp",
        type=float,
        default=0.5,
        help="The temperature applied pre-softmax. Default: 0.5",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="The temperature applied pre-softmax. Default: 0.7",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="The number of highest probability vocabulary tokens to keep for top-k sampling. Default: 50",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="The cumulative probability for nucleus sampling. Default: 0.9",
    )

    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="The parameter for repetition penalty. 1.0 means no penalty. Default: 1.2",
    )

    args = parser.parse_args()

    return args


def get_evaluation_args():
    description = "Collects the arguments for evaluating the model on any task"

    parser = argparse.ArgumentParser(description=description)

    # These defaults should be mostly fine

    parser.add_argument(
        "--seqlen", type=int, default=512, help="Sequence length. Default: 512"
    )

    parser.add_argument(
        "--max_iters", type=int, default=3, help="Number of iterations. Default: 3"
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=50304,
        help="Number of target classes OR vocab size. Default: 50304",
    )

    parser.add_argument(
        "--num_tokens",
        type=int,
        default=64,
        help="Number of output tokens to generate. Default: 64",
    )

    # Likely need to be overwritten

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./",
        help="Path for the model checkpoint. Default: ./",
    )

    parser.add_argument(
        "--n_heads", type=int, default=4, help="Number of attention heads. Default: 4"
    )

    parser.add_argument(
        "--width", type=int, default=384, help="Width dimension. Default: 384"
    )

    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Train baseline vanilla transformer model. Default: False",
    )

    parser.add_argument(
        "--num_blocks",
        type=int,
        default=1,
        help="Number of attention blocks. Default: 1",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        help="Strategy to use when inferencing. DDP should be enough",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="hellaswag",
        help="Which task to evaluate on. See full list on the EAI LM Eval Harness' GitHub repo",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
