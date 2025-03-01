import logging
import os
from argparse import Namespace
from typing import Any, Callable, Optional
import wandb

class UnifiedLogger:
    '''
    Holds both WandB and python logger objects
    '''
    def __init__(self, args, level: str = 'DEBUG') -> None:
        self.level = level

    def my_logger(self) -> logging.Logger:
        # Create logger
        logger = logging.getLogger('ReAct')
        logger.setLevel(self.level)

        # Create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')

        # Add formatter to ch
        ch.setFormatter(formatter)

        # Add ch to logger
        logger.addHandler(ch)

        return logger

    def wandb_logger(self, args: Any):
        key = os.environ.get("WANDB_API_KEY")
        wandb.login(key=key)

        wandb_id = None

        if isinstance(args.resume, str):
            # args.resume is of the form: "neel/ReAct_Jax/lxxn0x54 + 20"
            # we want to extract the run id, i.e "lxxn0x54"
            wandb_id = args.resume.split("+")[0].split("/")[-1].strip()

        wandb.init(
            project="ReAct_Jax",
            config=args,
            group=args.group,
            mode="online" if args.exp_logging else "offline",
            resume="allow",
            id=wandb_id,
            reinit=True,
            allow_val_change=True
        )

        wandb.run.log_code(
            "../",
            include_fn=lambda path: path.endswith(".py")
            or path.endswith(".ipynb")
            or path.endswith(".sh"),
        )

        return wandb

    def update_args_for_hypertuning(self, args: Namespace, experiment: Optional[Callable] = None):
        '''
        Consumes the experiment object provided by init_hypertuning() and updates the args dict
        '''
        arglist = ['lr', 'drop_rate', 'weight_decay', 'grad_clip', 'warmup_steps']

        for arg_name in arglist:
            setattr(args, arg_name, experiment.config[arg_name])

        args.epochs = 1 # for faster training

        return args

    def init_wandb_sweep(self) -> str:
        '''
        Deprecated. Use `optuna`.
        Setup Wandb Seep configs. Only run after wandb_logger() has been called
        '''
        sweep_configuration = {
            "method": "bayes",
            "name": "sweep",
            "metric": {"goal": "minimize", "name": "Train/loss"},
            "parameters": {
                "lr": {"max": 8e-2, "min": 1e-4},
                "drop_rate": {"max": 0.2, "min": 0.0},
                "weight_decay": {"max": 1e-3, "min": 1e-5},
                "grad_clip": {"max": 1.0, "min": 0.1},
                "warmup_steps": {"values": list(range(0, 1000, 100))},
            },
            "early_terminate": {"type": "hyperband", "max_iter": 48, "s": 4},
        }

        sweep_id = wandb.sweep(
            sweep=sweep_configuration, project="ReAct_Jax", entity="neel"
        )

        return sweep_id
