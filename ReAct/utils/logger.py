import logging
import wandb
import os

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
    
    def wandb_logger(self, args):
        key = os.environ.get('WANDB_API_KEY')
        wandb.login(key=key)
        
        if args.resume:
            id = args.resume.split('/')[2]
        else:
            id = None
            
        wandb.init(project='ReAct_Jax', config=args, anonymous='allow',
                   mode=args.wandb, magic=True, resume=args.resume, id=id)
        
        wandb.run.log_code(
            "../",
            include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".sh"))
        
        return wandb

if __name__ == '__main__':
    my_logger = UnifiedLogger(level='INFO').logger

    my_logger.debug('debug message')
    my_logger.info('info message')
    my_logger.warning('warn message')
    my_logger.error('error message')
    my_logger.critical('critical message')