import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as sharding
from jax import config
from jaxtyping import PRNGKeyArray

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
    
    # ========= Logging =========
        
    logger = UnifiedLogger(args, level='DEBUG')
    
    _, model_key = jax.random.split(key)
    
    train_dataset = TinyStoriesDataset(split='train', max_length=args.seqlen, bsz=args.batch_size)
    trainloader = train_dataset.create_dataloader()
    
    trainloader.shuffle = True
    
    valloader = TinyStoriesDataset(
        split='validation', max_length=args.seqlen, bsz=args.batch_size).create_dataloader()
    
    num_devices = jax.local_device_count()
    print(f'Number of devices: {num_devices}')
    
    devices = mesh_utils.create_device_mesh((num_devices, 1))
    shard = sharding.PositionalSharding(devices)
    
    trainer = Trainer(args, logger, shard)
    trainer.train(args.epochs, trainloader, valloader, key, train_dataset.mask_tokens)

if __name__ == '__main__':
    key = jax.random.PRNGKey(69)
    main(key)