import os
import equinox as eqx
import jax
import jax.numpy as jnp

from jax import tree_util as jtu
from jaxtyping import Array, PRNGKeyArray
from typing import Optional, Callable

def calc_performance_metrics(fn: Callable, static_argnums: tuple[int], args: tuple[int]) -> float:
    '''
    Calculate the number of FLOPs and memory requirements
    for a given function using AOT compilation.
    Returns the number of FLOPs in PetaFLOPs
    '''
    compiled = jax.jit(fn, static_argnums=static_argnums).lower(*args).compile()
    cost_anal = compiled.cost_analysis()
    
    return cost_anal[0]['flops'] / 1e15
    
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