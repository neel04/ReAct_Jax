import math
import os
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jaxtyping import Array, PRNGKeyArray

def convert_flops(params: int) -> str:
    if params == 0:
        return "0"
    
    size_name = ("", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs", "EFLOPs", "ZFLOPs", "YFLOPs")
    i = int(math.floor(math.log(params, 1000)))
    p = math.pow(1000, i)
    s = round(params / p, 2)
    
    return "%s %s" % (s, size_name[i])

def calc_performance_metrics(args, my_logger: Callable) -> None:
    '''
    Estimates FLOPs consumed during a single fwd + bwd pass.
    Taken from EleutherAI's GPT-NeoX repo: https://rb.gy/33d6zg
    
    Returns: the total number of FLOPs
    '''
    iter_factor = 3
    args.tokens = args.batch_size * args.seqlen
    args.kv_size_ratio = 1
    
    # TODO: Ignores activation checkpointing. Fix this at some point 
    my_logger.warning('! Ignoring activation checkpointing in FLOPs calculation !')
        
    qkv_flops = int(iter_factor * 2 * (1 + 2 * args.kv_size_ratio) * args.num_classes * args.tokens * args.width * args.width)
    attention_matrix_flops = iter_factor * 2 * args.num_classes * args.tokens * args.seqlen * args.width
    attention_over_values_flops = iter_factor * 2 * args.num_classes * args.tokens * args.seqlen * args.width
    linear_projection_flops = iter_factor * 2 * args.num_classes * args.tokens * args.width * args.width
    ffn_flops = iter_factor * 16 * args.num_classes * args.tokens * args.width * args.width
    
    # handle NewGELU
    ffn_flops *= 3.75
    
    embedding_flops = 6 * args.tokens * args.width * args.num_classes
    total_flops = qkv_flops + attention_matrix_flops + attention_over_values_flops + linear_projection_flops + ffn_flops + embedding_flops
    my_logger.info(f"Total FLOPs for the Model: {convert_flops(total_flops)} for a single fwd + bwd pass\n")
    
def half_precision(model: eqx.Module) -> eqx.Module:
    return jtu.tree_map(lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x, model)

def save_eqx_obj(save_dir: str, filename: str, obj: tuple):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    eqx.tree_serialise_leaves(filename, obj)
        
def load_eqx_obj(filepath: str, obj: tuple) -> tuple:
    return eqx.tree_deserialise_leaves(path_or_file=filepath,
                                       like=obj)

def broad_to_bsz(arr: Array, shape: tuple) -> Array:
    return jnp.broadcast_to(arr, shape)

def count_params(model: eqx.Module) -> int:
    params_fn = lambda model: sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))  # noqa: E731
    num_params, non_embed_params = params_fn(model), params_fn(model.main_block)
    
    num_params /= 1_000_000
    non_embed_params /= 1_000_000
    
    print(f"\nModel # of parameters: {num_params:.2f}M\n# of recurrent parameters: {non_embed_params:.2f}M\n")
    
    return num_params

def get_rand_nums(key: PRNGKeyArray, lower_bound: int, upper_bound: int, bsz: int, bias_val: Optional[int] = None) -> Array:
    '''
    Generate random numbers from a uniform distribution
    or bias it towards a certain value, if provided
    '''
    if bias_val is None:
        dist = jax.random.randint(key, shape=(bsz,), minval=lower_bound, maxval=upper_bound)
    else:
        dist = jnp.clip(jax.random.normal(key, (bsz,)) * (bias_val ** .5) + bias_val + 1, lower_bound, upper_bound)
        
    return dist.astype(int)

@jax.jit
def inverted_freq(arr: Array):
    arr = arr.sort(0)
    
    values, counts = jnp.unique(arr,
                                return_counts=True,
                                size=64)
    
    # Replace 0s with any element for scaling to work
    counts = jnp.where(counts == 0, counts[0], counts)
    
    inv_weights = (counts.max() / counts) # scale it down
    
    return inv_weights[arr - arr.min()]