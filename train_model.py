import jax

from jax import config
from jaxtyping import PRNGKeyArray
from torch.utils.data import DataLoader

from ReAct.data.reverse_string import RevDataset
from ReAct.utils.arg_parser import parse_args
from ReAct.utils.logger import UnifiedLogger
from ReAct.utils.trainer import Trainer

def main(key: PRNGKeyArray):
    args = parse_args()
    
    # Enter debugging mode, disabling JIT
    if args.debug:
        config.update("jax_debug_nans", True)
        config.update("jax_debug_infs", True)
        config.update("jax_disable_jit", True)
    
    # ========= Logging =========
        
    logger = UnifiedLogger(args, level='DEBUG')
    
    _, model_key = jax.random.split(key)

    dataset = RevDataset(args.seqlen, 3, args.dataset_length)

    # Truncated dataset for calculating training accuracy
    trunc_dataset = RevDataset(args.seqlen, 3, args.dataset_length // 20)
    val_dataset = RevDataset(args.seqlen, args.seqlen, args.dataset_length // 20)

    trainloader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             drop_last=True,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True,
                             prefetch_factor=4)

    truncloader = DataLoader(trunc_dataset,
                             batch_size=args.batch_size,
                             drop_last=True,
                             shuffle=False)

    valloader = DataLoader(val_dataset,
                           batch_size=args.batch_size,
                           drop_last=True,
                           shuffle=False)


    trainer = Trainer(args, model_key, logger)
    trainer.train(args.num_epochs, trainloader, truncloader, valloader)

if __name__ == '__main__':
    key = jax.random.PRNGKey(69)
    main(key)