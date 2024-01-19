from typing import Callable, Tuple

import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as sharding
from jax import config
from jaxtyping import PRNGKeyArray

import wandb
from ReAct.data.tinystories import TinyStoriesDataset
from ReAct.utils.arg_parser import parse_args
from ReAct.utils.logger import UnifiedLogger
from ReAct.utils.trainer import Trainer


def main(key: PRNGKeyArray):
    args = parse_args()
    jax.config.update('jax_threefry_partitionable', True) # for parallelization
    
    # Enter debugging mode, disabling JIT
    if args.debug:
        config.update("jax_debug_nans", True)
        config.update("jax_debug_infs", True)
        config.update("jax_disable_jit", True)
    
    # ========= Data =========
    train_dataset = TinyStoriesDataset(split='train', max_length=args.seqlen, bsz=args.batch_size)
    trainloader = train_dataset.create_dataloader(args.debug)
    
    valloader = TinyStoriesDataset(
        split='validation', max_length=args.seqlen, bsz=args.batch_size).create_dataloader()
    
    # preshifting the datasets
    print('\nPre-processing the training dataset...\n')
    shift_fn = train_dataset.shift_tokens
    trainloader = list(trainloader) # list of dicts -> tuples
    trainloader = jax.tree_map(lambda x: shift_fn(x), trainloader)
    
    print('\nPre-processing the validation dataset...\n')
    valloader = list(valloader)
    valloader = jax.tree_map(lambda x: shift_fn(x), valloader)
    
    num_devices = jax.local_device_count()
    print(f'Number of devices: {num_devices}')
    
    devices = mesh_utils.create_device_mesh((num_devices, 1))
    shard = sharding.PositionalSharding(devices)
    
    if args.tune_hyperparams:
        logger = UnifiedLogger(args, level='DEBUG')
        sweep_id = logger.init_wandb_sweep()
        
        wandb.agent(sweep_id=sweep_id, count=50,
                    function=lambda: kickoff_sweeps(
                        args=args,
                        loggers=logger,
                        loaders=(trainloader, valloader),
                        decode_fn=train_dataset.tok.decode,
                        shard=shard,
                        key=key)
                    )
    else:
        logger = UnifiedLogger(args, level='DEBUG')
        my_logger, wandb_logger = logger.my_logger(), logger.wandb_logger(args)
        
        trainer = Trainer(args, logger=(my_logger, wandb_logger),
                            loaders=(trainloader, valloader),
                            decode_fn=train_dataset.tok.decode,
                            shard=shard,
                            key=key)
        
        trainer.train()

def kickoff_sweeps(args: dict, loggers: UnifiedLogger, loaders: Tuple,
                     decode_fn: Callable, shard: sharding.PositionalSharding,
                     key: PRNGKeyArray):
    
    my_logger = loggers.my_logger()
    trainloader, valloader = loaders
    decode_fn = decode_fn
    
    with wandb.init(project='ReAct_Jax', config=args, anonymous='allow',
                    entity='neel', group='Sweeps', mode=args.exp_logging) as wandb_logger:
        
        args = loggers.update_args_for_hypertuning(args, wandb)
        
        trainer = Trainer(args, logger=(my_logger, wandb_logger),
                            loaders=(trainloader, valloader),
                            decode_fn=decode_fn,
                            shard=shard,
                            key=key)
        
        trainer.train()

if __name__ == '__main__':
    key = jax.random.PRNGKey(69)
    main(key)