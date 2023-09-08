import argparse

def parse_args():
    description = "Training script - kicks off training of the model"
    epilog = "Hyperparameters defaults may not be optimal. Please use --help to check available options."

    parser = argparse.ArgumentParser(description=description, epilog=epilog)

    # Model
    parser.add_argument('--num_blocks', type=int, default=2,
                        help='Number of attention blocks. Default: 2')

    parser.add_argument('--width', type=int, default=512,
                        help='Width dimension. Default: 512')

    parser.add_argument('--drop_rate', type=float, default=0.1,
                        help='Dropout rate. Default: 0.1')

    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of target classes. Default: 2')

    parser.add_argument('--max_iters', type=int, default=10,
                        help='Number of iterations. Default: 10')

    parser.add_argument('--seqlen', type=int, default=16,
                        help='Sequence length. Default: 16')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size. Default: 4')

    parser.add_argument('--lr', type=float, default=3e-3,
                        help='Learning rate. Default: 1e-3')

    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay. Default: 1e-4')

    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps. Default: 1000')

    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping. Default: 1.0')

    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs. Default: 100')

    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save interval. Default: 10')

    parser.add_argument('--save_dir', type=str, default='/Users/neel/Documents/research/ReAct_Jax/ReAct/outputs/',
                        help='Save directory for checkpoints. Default: ./outputs/. Keep the slash at the end')

    parser.add_argument('--dataset_length', type=int, default=50_000,
                        help='Length of the dataset. Default: 50_000')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)