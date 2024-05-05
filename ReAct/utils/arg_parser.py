import argparse

def parse_args():
    description = "Training script - kicks off training of the model"
    epilog = "Hyperparameters defaults may not be optimal. Please use --help to check available options."

    parser = argparse.ArgumentParser(description=description, epilog=epilog)

    # Model
    parser.add_argument('--num_blocks', type=int, default=1,
                        help='Number of attention blocks. Default: 1')
    
    parser.add_argument('--dataset', type=str, default='openwebtext',
                        help='Dataset to use. Default: openwebtext')

    parser.add_argument('--width', type=int, default=384,
                        help='Width dimension. Default: 384')

    parser.add_argument('--drop_rate', type=float, default=0.1,
                        help='Dropout rate. Default: 0.1')

    parser.add_argument('--num_classes', type=int, default=50304,
                        help='Number of target classes OR vocab size. Default: 50304')
    
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads. Default: 4')

    parser.add_argument('--max_iters', type=int, default=10,
                        help='Number of iterations. Default: 10')

    parser.add_argument('--seqlen', type=int, default=512,
                        help='Sequence length. Default: 512')

    parser.add_argument('--batch_size', type=int, default=208,
                        help='Batch size. Default: 208')

    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Learning rate. Default: 3e-5')
    
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay. Default: 1e-5')
    
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps. Default: 100')
    
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='Number of gradient accumulation steps. Default: 1')

    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping. Default: 1.0')

    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs. Default: 1')
    
    parser.add_argument('--beta_1', type=float, default=0.9,
                        help='Adam beta 1. Default: 0.9')
    
    parser.add_argument('--beta_2', type=float, default=0.999,
                        help='Adam beta 2. Default: 0.999')

    parser.add_argument('--save_interval', type=int, default=2500,
                        help='Save interval. Default: 2500')
    
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='Log interval. Default: 1000')

    parser.add_argument('--save_dir', type=str, default='/Users/neel/Documents/research/ReAct_Jax/ReAct/outputs/',
                        help='Save directory for checkpoints. Default: ./outputs/. Keep the slash at the end')

    parser.add_argument('--debug', action='store_true', default=False,
                        help='Debug mode - disables JIT. Default: False')
    
    parser.add_argument('--bf16', action='store_true', default=False,
                        help='Enable mixed precision training. Default: False')
    
    parser.add_argument('--exp_logging', action='store_const', const='online', default='disabled',
                    help='Enable experiment logging in the cloud. Default: disabled')
    
    parser.add_argument('--group', type=str, default='OpenWebText',
                        help='WandB group name. Default: OWT')
    
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint. Default: None')
    
    parser.add_argument('--prompt', type=str, default=None,
                        help='input for inference. Default: None')
    
    parser.add_argument('--resume', type=str, default=None,
                        help='Obtain WandB run_path from Overview tab and append the'
                             'step number. \nExample arg: "neel/ReAct_Jax/6ktmhalt/ + 200"')
    
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='Train baseline vanilla transformer model. Default: False')
    
    parser.add_argument('--tune_hyperparams', action='store_true', default=False,
                        help='Tune hyperparameters using wandb sweep. Default: False')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)