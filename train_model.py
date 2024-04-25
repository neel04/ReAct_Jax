import platform
import jax

if platform.processor() != 'arm':
    jax.distributed.initialize() # don't run on apple sillicon

import optuna
from jax import config
from jax.experimental.compilation_cache import compilation_cache
from jaxtyping import PRNGKeyArray
from optuna.integration.wandb import WeightsAndBiasesCallback

from ReAct.data.minipile import MiniPileDataset
from ReAct.data.owt import OpenWebTextDataset
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
        config.update("jax_disable_jit", False)
        config.update("jax_default_matmul_precision", "bfloat16")

    # ========= Data =========
    match args.dataset.lower():
        case 'tinystories':
            dataset = TinyStoriesDataset
        case 'owt':
            dataset = OpenWebTextDataset
        case 'minipile':
            dataset = MiniPileDataset

    train_dataset = dataset(split='train', max_length=args.seqlen, bsz=args.batch_size)
    val_dataset = dataset(split='test', max_length=args.seqlen, bsz=args.batch_size)

    # ========= Training/Hypertuning =========

    if args.tune_hyperparams:
        args.group = 'Sweeps' if args.baseline else 'Sweeps_5i'
        
        trainloader = train_dataset.create_dataloader('20%')
        valloader = val_dataset.create_dataloader('20%')

        # Create optuna hypertununing study
        study = optuna.create_study(
            direction="minimize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(
                seed=69,
                consider_magic_clip=True,
                consider_endpoints=True,
                n_startup_trials=5,
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=200, n_min_trials=10
            ),
        )

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
            metric_name='Train/acc',
            wandb_kwargs=wandb_kwargs,
            as_multirun=True
        )

        study.optimize(
            lambda trial: kickoff_optuna(trial=trial, **trainer_kwargs),
            n_trials=50,
            callbacks=[wandbc]
        )
        
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html("optuna_plot.html")
        
        print(f"Best trial: {study.best_trial}")
        print(f'\nValue: {study.best_trial.value}\nParams: {study.best_trial.params}\n')

    else:
        trainloader = train_dataset.create_dataloader()
        valloader = val_dataset.create_dataloader()

        logger = UnifiedLogger(args, level='DEBUG')
        my_logger, wandb_logger = logger.my_logger(), logger.wandb_logger(args)

        trainer = Trainer(args, logger=(my_logger, wandb_logger),
                            loaders=(trainloader, valloader),
                            decode_fn=train_dataset.tok.decode,
                            key=key)

        my_logger.info(f"# of all devices: {jax.device_count()}")
        my_logger.info(f"# of hosts: {jax.process_count()}")
        my_logger.info(f"Host id: {jax.process_index()}")

        with jax.spmd_mode('allow_all'):
            trainer.train()

def kickoff_optuna(trial, **trainer_kwargs):
    args = trainer_kwargs['args']

    args.epochs = 2

    args.lr = trial.suggest_float('lr', 1e-4, 1e-3, step=1e-4)
    args.drop_rate = trial.suggest_float('drop_rate', 0.0, 0.1, step=0.02)
    args.weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, step=2e-4)
    args.warmup_steps = trial.suggest_int('warmup_steps', 0, 500, step=100)

    args = trainer_kwargs['args']

    # ========= Logging ========
    logger = UnifiedLogger(args, level='DEBUG')
    my_logger, wandb_logger = logger.my_logger(), logger.wandb_logger(args)
    trainer_kwargs['logger'] = (my_logger, wandb_logger)

    trainer = Trainer(**trainer_kwargs)
    
    my_logger.info(f"# of all devices: {jax.device_count()}")
    my_logger.info(f"# of hosts: {jax.process_count()}")
    my_logger.info(f"Host id: {jax.process_index()}")

    with jax.spmd_mode('allow_all'):
        loss = trainer.train(trial)
    
    return loss

if __name__ == '__main__':
    compilation_cache.initialize_cache('./compilation_cache')
    key = jax.random.PRNGKey(69)
    main(key)
