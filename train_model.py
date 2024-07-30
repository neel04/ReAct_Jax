import platform
import jax

jax.config.update("jax_compilation_cache_dir", "./ReAct/compilation_cache")

if platform.processor() != 'arm':
    jax.distributed.initialize() # don't run on apple sillicon

import optuna
from jax import config
from jaxtyping import PRNGKeyArray
from optuna.integration.wandb import WeightsAndBiasesCallback

from ReAct.data.gh_code import GithubCodeDataset
from ReAct.data.minipile import MiniPileDataset
from ReAct.data.owt import OpenWebTextDataset
from ReAct.data.tinystories import TinyStoriesDataset
from ReAct.utils.arg_parser import parse_args
from ReAct.utils.logger import UnifiedLogger
from ReAct.utils.trainer import Trainer

# ruff: noqa: E402, E731

def main(key: PRNGKeyArray):
    args = parse_args()
    config.update('jax_threefry_partitionable', True) # for parallelization
    
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
        case 'github':
            dataset = GithubCodeDataset

    train_dataset = dataset(split='train', max_length=args.seqlen, bsz=args.batch_size)
    val_dataset = dataset(split='test', max_length=args.seqlen, bsz=args.batch_size)

    # ========= Training/Hypertuning =========
    init_hyperparams = [
        {"lr": 7e-4, "drop_rate": 0.02, "weight_decay": 1e-4, "warmup_steps": 170, "beta_1": 0.95, "beta_2": 0.98, "nesterov": False},
        {"lr": 1e-4, "drop_rate": 0.00, "weight_decay": 1e-4, "warmup_steps": 200, "beta_1": 0.95, "beta_2": 0.99, "nesterov": True}
    ]

    if args.tune_hyperparams:
        args.group = 'Sweeps_base' if args.baseline else f'Sweeps_{args.max_iters}i'

        trainloader = train_dataset.create_dataloader(':10%')
        valloader = val_dataset.create_dataloader()
        jax.experimental.multihost_utils.sync_global_devices('Sync up dataset preprocessing.')

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
                n_startup_trials=5,
                n_min_trials=5
            )
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
            metric_name='Val/loss',
            wandb_kwargs=wandb_kwargs,
            as_multirun=True
        )

        # enqueue a few handpicked hyperparams for trials
        [study.enqueue_trial(hyperparams) for hyperparams in init_hyperparams]

        study.optimize(
            lambda trial: kickoff_optuna(trial=trial, **trainer_kwargs),
            n_trials=50,
            callbacks=[wandbc],
        )

        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html("optuna_plot.html")
        
        print(f"Best trial: {study.best_trial}")
        print(f'\nValue: {study.best_trial.value}\nParams: {study.best_trial.params}\n')

    else:
        trainloader = train_dataset.create_dataloader(":99%")
        valloader = val_dataset.create_dataloader()
        jax.experimental.multihost_utils.sync_global_devices('Sync up dataset preprocessing.')

        logger = UnifiedLogger(args, level="DEBUG")
        my_logger, wandb_logger = logger.my_logger(), logger.wandb_logger(args)

        trainer = Trainer(
            args,
            logger=(my_logger, wandb_logger),
            loaders=(trainloader, valloader),
            decode_fn=train_dataset.tok.decode,
            key=key,
        )

        my_logger.info(f"# of all devices: {jax.device_count()}")
        my_logger.info(f"# of hosts: {jax.process_count()}")
        my_logger.info(f"Host id: {jax.process_index()}")

        with jax.spmd_mode('allow_all'):
            trainer.train()

def kickoff_optuna(trial, **trainer_kwargs):
    args = trainer_kwargs['args']

    args.epochs = 2
    
    # Regularization hyperparams
    args.lr = trial.suggest_float('lr', 1e-6, 1e-2)
    args.drop_rate = trial.suggest_float('drop_rate', 0.0, 0.05, step=0.01)
    args.weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3)
    args.warmup_steps = trial.suggest_int('warmup_steps', 0, 300, step=10)

    # Optimizer hyperparams
    args.beta_1 = trial.suggest_categorical('beta_1', [0.8, 0.85, 0.9, 0.95, 0.98, 0.99])
    args.beta_2 = trial.suggest_categorical('beta_2', [0.9, 0.95, 0.98, 0.99, 0.999])
    args.nesterov = trial.suggest_categorical('nesterov', [True, False])

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
        output = trainer.train(trial)
    
    return jax.numpy.nan_to_num(output, nan=9e9)

if __name__ == '__main__':
    key = jax.random.PRNGKey(69)
    main(key)
    exit(0)