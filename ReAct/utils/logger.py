import logging
import wandb

class UnifiedLogger:
    '''
    Holds both WandB and python logger objects
    '''
    def __init__(self, args, level: str = 'DEBUG', mode: str = 'disabled') -> None:
        self.mode = mode
        
        self.logger = self.my_logger(level)
        self.wandb_logger = self.wandb_logger(args)

    def my_logger(self, level: str) -> logging.Logger:
        # Create logger
        logger = logging.getLogger('ReAct')
        logger.setLevel(level)

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
    
    def wandb_logger(self, args):
        wandb.login(key="78c7285b02548bf0c06dca38776c08bb6018593f")
        
        wandb.init(project='ReAct_Jax', magic=True, anonymous='allow',
                   mode=self.mode, config=args)
        
        self.wandb.run.log_code( # type: ignore
            "/workspace/", 
            include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".sh"))
        
        return wandb

if __name__ == '__main__':
    my_logger = UnifiedLogger(level='INFO').logger

    my_logger.debug('debug message')
    my_logger.info('info message')
    my_logger.warning('warn message')
    my_logger.error('error message')
    my_logger.critical('critical message')