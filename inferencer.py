import os

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from ReAct.model.react import React
from ReAct.utils.arg_parser import parse_args
from ReAct.utils.helpers import convert_to_jax, count_params
from ReAct.utils.logger import UnifiedLogger


class Inferencer:
    def __init__(self, args, logger):
        self.key = jax.random.PRNGKey(69)
        self.logger = logger.my_logger()
        
        for k, v in vars(args).items():
            setattr(self, k, v)

    def skeleton_model(self, key: PRNGKeyArray):
        model = React(
            self.seqlen, self.max_iters, self.num_blocks, self.width,
            self.drop_rate, self.num_classes, key
        )
        
        return model
    
    def load_eqx_model(self, filepath, key: PRNGKeyArray):
        with open(filepath, 'rb') as f:
            model = self.skeleton_model(key)
            return eqx.tree_deserialise_leaves(f, model)
    
    def encode_input(self, my_input: str):
        encoded = [int(i) for i in my_input]
        
        if len(encoded) < self.seqlen:
            self.logger.info('Input is too short, padding with 0s')
            encoded = encoded + [0] * (self.seqlen - len(encoded))
        
        assert all([i in [0, 1] for i in encoded]), "Input must be binary"
        assert len(encoded) == self.seqlen, "BUG: Input must be of length --seqlen"
        
        return encoded
        
    
    def inference(self, my_input: str):
        model = self.load_eqx_model(self.checkpoint_path, self.key)
        count_params(model)
        
        # Convert to JAX
        my_input = convert_to_jax(my_input)
        
        # Make prediction
        prediction = model(my_input, self.max_iters, None, False).argmax(-1)
        
        return [i.item() for i in prediction]

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
    print(f'Correct Output: {prompt[::-1]}')
    print(f'\nAccuracy: {sum([1 for i, j in zip(prompt[::-1], output) if i == j]) / len(prompt) * 100}%')
    print(f"\n{'~' * 50}\n")