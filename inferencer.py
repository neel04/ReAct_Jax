import os

import jax
from jaxtyping import PRNGKeyArray
from icecream import ic

from ReAct.data.tinystories import TinyStoriesDataset
from ReAct.model.react import React
from ReAct.utils.arg_parser import parse_args
from ReAct.utils.helpers import convert_to_jax, count_params, load_eqx_obj
from ReAct.utils.logger import UnifiedLogger
from ReAct.utils.trainer import Trainer

class Inferencer:
    def __init__(self, args, logger):
        self.dummy_dataset = TinyStoriesDataset(split='train', max_length=args.seqlen, bsz=args.batch_size)
        self.trainer = Trainer(args, logger, self.dummy_dataset.tok.decode, self.dummy_dataset.shift_tokens)
        
        self.key = jax.random.PRNGKey(69)
        self.logger = logger.my_logger()
        
        for k, v in vars(args).items():
            setattr(self, k, v)

    def skeleton_model(self, key: PRNGKeyArray):
        model = React(self.n_heads, self.seqlen, self.max_iters, self.num_blocks,
                      self.width, self.drop_rate, self.num_classes, key)
        
        return model
    
    def encode_input(self, my_input: str):
        encoded = self.dummy_dataset.tok.encode(my_input).ids[:-1]
        # Remove all 0 and 2 elements, which are padding and EOS tokens
        encoded = [i for i in encoded if i not in [0, 2]]
        return encoded
    
    def inference(self, my_input: str, num_tokens: int = 24):
        model = self.skeleton_model(self.key)
        model = load_eqx_obj(self.checkpoint_path, model)
        
        count_params(model)
        
        # Convert to JAX
        my_input = convert_to_jax(my_input)
        
        # Make prediction
        output = self.trainer.generate(model, my_input, num_tokens, temperature=0.35)
        
        return output
    
if __name__ == '__main__':
    
    args = parse_args()
    logger = UnifiedLogger(args, level='DEBUG')
    my_logger = logger.my_logger()
    
    my_logger.info('Make sure to provide the correct args per the model file!')
    my_logger.info('These are: seqlen | max_iters | num_blocks | width |  num_classes | prompt')
    my_logger.info("Check them from the WandB runs")
    print(f"{'-'*50}\n")
    
    assert args.checkpoint_path is not None, "Please provide a checkpoint path"
    assert os.path.exists(args.checkpoint_path), "Please provide a valid checkpoint path | File does not exist"
    assert args.prompt is not None, "Please provide a prompt/input for inference"
    
    # Make sure the user has specified these arguments
    inferencer = Inferencer(args, logger)
    prompt = inferencer.encode_input(args.prompt)
    
    output = inferencer.inference(prompt)
    
    print(f"\n{'~' * 50}\n")
    print(f'Input:          {prompt}')
    print(f'Model Output:   {output}')
    print(f"\n{'~' * 50}\n")