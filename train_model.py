import jax
import equinox as eqx
import torch.utils.data.dataloader as DataLoader

from jaxtyping import PRNGKeyArray
from tqdm import tqdm

from ReAct.utils.helpers import convert_to_jax
from ReAct.utils.logger import UnifiedLogger
from ReAct.utils.arg_parser import parse_args
from ReAct.utils.dataset import RevDataset
from ReAct.utils.trainer import Trainer

def main(key: PRNGKeyArray, model: eqx.Module):
    args = parse_args()
    logger = UnifiedLogger(level='DEBUG', mode='disabled')
    logger.log_hyperparams(args)
    
    _, model_key = jax.random.split(key)

    dataset = RevDataset(args.seqlen, 3, args.dataset_length)

    # Truncated dataset for calculating training accuracy
    trunc_dataset = RevDataset(args.seqlen, 3, args.dataset_length // 20)
    val_dataset = RevDataset(args.seqlen, args.seqlen, args.dataset_length // 20)

    trainloader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             drop_last=True,
                             shuffle=False)

    truncloader = DataLoader(trunc_dataset,
                             batch_size=args.batch_size,
                             drop_last=True,
                             shuffle=False)

    valloader = DataLoader(val_dataset,
                           batch_size=args.batch_size,
                           drop_last=True,
                           shuffle=False)


    trainer = Trainer(args, model_key, logger)

    for epoch in range(args.num_epochs):

    return model
     
if __name__ == '__main__':
    main()