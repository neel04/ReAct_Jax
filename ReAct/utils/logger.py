import logging
import os
from typing import Callable, Generator

import comet_ml
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
    
    def wandb_logger(self, args: dict):
        key = os.environ.get('WANDB_API_KEY')
        wandb.login(key=key)
        
        if args.resume:
            # args.resume is of the form: "neel/ReAct_Jax/lxxn0x54 + 20"
            # we want to extract the run id, i.e "lxxn0x54"
            id = args.resume.split("+")[0].split("/")[-1].strip()
        else:
            id = None
            
        wandb.init(project='ReAct_Jax', config=args, anonymous='allow', group='TinyStories',
                   mode=args.exp_logging, resume='allow', id=id)
        
        wandb.run.log_code(
            "../",
            include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".sh"))
        
        return wandb

    def comet_ml_logger(self, args: dict):
        key = os.environ.get('COMET_API_KEY')
        
        comet_ml.init(project_name='react-jax')
        
        experiment = comet_ml.Experiment(api_key=key, disabled=args.exp_logging,
                                         auto_metric_step_rate=5, auto_histogram_weight_logging=True,
                                         auto_histogram_gradient_logging=True, auto_histogram_activation_logging=True)
        
        experiment.log_parameters(vars(args))
        
        return experiment
    
    def init_hypertuning(self, args: dict) -> Generator:
        config = {"algorithm": "bayes",
            "spec": {
                "maxCombo": 0,
                "metric": "Train/acc",
                "objective": "maximize",
            },
            
            "parameters": {
                "lr": {"type": "float", "min": 1e-6, "max": 1e-3},
                "drop_rate": {"type": "float", "min": 0.0, "max": 0.2},
                "weight_decay": {"type": "float", "min": 1e-5, "max": 1e-3},
                "grad_clip": {"type": "float", "min": 0.1, "max": 1.0},
                "warmup_steps": {"type": "discrete", "values": list(range(0, 1000, 100))},
            },
            
            "name": "My Bayesian Search",
            "trials": 50,
        }
        
        opt = comet_ml.Optimizer(config, api_key=os.environ.get('COMET_API_KEY'))
        
        return opt.get_experiments(project_name="react-jax-sweep")
    
    def update_args_for_hypertuning(self, args: dict, experiment: Callable = None):
        '''
        Consumes the experiment object provided by init_hypertuning() and updates the args dict
        '''
        arglist = ['lr', 'drop_rate', 'weight_decay', 'grad_clip', 'warmup_steps']
        
        for arg_name in arglist:
            # args.arg_name = experiment.config.arg_name
            setattr(args, arg_name, experiment.config[arg_name])
        
        args.epochs = 2 # for faster training
        
        return args
    
    def init_wandb_sweep(self) -> int:
        '''
        Setup Wandb Seep configs. Only run after wandb_logger() has been called
        '''
        sweep_configuration = {
            "method": "random",
            "name": "sweep",
            "metric": {"goal": "maximize", "name": "Train/acc"},
            "parameters": {
                "lr": {"max": 1e-2, "min": 1e-5},
                "drop_rate": {"max": 0.2, "min": 0.0},
                "weight_decay": {"max": 1e-3, "min": 1e-5},
                "grad_clip": {"max": 1.0, "min": 0.1},
                "warmup_steps": {"values": list(range(0, 1000, 100))},
            },
        }
        
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="ReAct_Jax",
                               entity='neel')
        
        return sweep_id
        
if __name__ == '__main__':
    my_logger = UnifiedLogger(level='INFO').logger

    my_logger.debug('debug message')
    my_logger.info('info message')
    my_logger.warning('warn message')
    my_logger.error('error message')
    my_logger.critical('critical message')