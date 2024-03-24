import platform
import jax

if platform.processor() != 'arm':
    jax.distributed.initialize() # don't run on apple sillicon

import jax.numpy as jnp
import optuna
from jax import config
from jaxtyping import PRNGKeyArray
from optuna.integration.wandb import WeightsAndBiasesCallback

from ReAct.data.owt import OpenWebTextDataset
from ReAct.data.tinystories import TinyStoriesDataset
from ReAct.data.minipile import MiniPileDataset
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
        config.update("jax_disable_jit", False)
    
    # ========= Data =========
    #TODO: Use inheritance to avoid seperate file for each dataset
    
    match args.dataset.lower():
        case 'tinystories':
            dataset = TinyStoriesDataset
        case 'owt':
            dataset = OpenWebTextDataset
        case 'minipile':
            dataset = MiniPileDataset
    
    train_dataset = dataset(split='train', max_length=args.seqlen, bsz=args.batch_size)
    val_dataset = dataset(split='test', max_length=args.seqlen, bsz=args.batch_size)
    
    trainloader = train_dataset.create_dataloader()
    valloader = val_dataset.create_dataloader()
    
    # ========= Training/Hypertuning =========
    
    if args.tune_hyperparams:
        args.group = 'Sweeps' if args.baseline else 'Sweeps_5i'
        
        study = optuna.create_study(direction='minimize',
                                    study_name='ReAct_Jax',
                                    load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(
                                        seed=69,
                                        consider_magic_clip=True,
                                    ))
        
        wandb_kwargs = {
            "project": "ReAct_Jax",
            "config": args,
            "anonymous": "allow",
            "entity": "neel",
            "magic": True,
        }
        
        trainer_kwargs = {
            "args": args,
            "loaders": (trainloader, valloader),
            "decode_fn": train_dataset.tok.decode,
            "key": key
        }
        
        wandbc = WeightsAndBiasesCallback(
            metric_name='Train/loss',
            wandb_kwargs=wandb_kwargs,
            as_multirun=True
        )
        
        study.optimize(lambda trial: kickoff_optuna(trial=trial, **trainer_kwargs), n_trials=50, callbacks=[wandbc])

    else:
        logger = UnifiedLogger(args, level='DEBUG')
        my_logger, wandb_logger = logger.my_logger(), logger.wandb_logger(args)
        
        trainer = Trainer(args, logger=(my_logger, wandb_logger),
                            loaders=(trainloader, valloader),
                            decode_fn=train_dataset.tok.decode,
                            key=key)
        
        my_logger.info(f"# of all devices: {jax.device_count()}")
        my_logger.info(f"# of hosts: {jax.process_count()}")
        
        with jax.spmd_mode('allow_all'):
            trainer.train()

def kickoff_optuna(trial, **trainer_kwargs):
    args = trainer_kwargs['args']
    
    args.epochs = 2
    
    args.lr = trial.suggest_float('lr', 1e-4, 1e-2)
    args.drop_rate = trial.suggest_float('drop_rate', 0.0, 0.2)
    args.weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3)
    args.warmup_steps = trial.suggest_int('warmup_steps', 0, 2000, step=100)
    
    args = trainer_kwargs['args']
     
    # ========= Logging ========
    logger = UnifiedLogger(args, level='DEBUG')
    my_logger, wandb_logger = logger.my_logger(), logger.wandb_logger(args)
    trainer_kwargs['logger'] = (my_logger, wandb_logger)
    
    trainer = Trainer(**trainer_kwargs)
    
    with jax.spmd_mode('allow_all'):
        loss = trainer.train()
        
    return jnp.nan_to_num(loss, nan=9999.0) # return the loss

if __name__ == '__main__':
    key = jax.random.PRNGKey(69)
    main(key)