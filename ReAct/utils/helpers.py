import os

import equinox as eqx
import jax
import jax.numpy as jnp
import torch
from jax import tree_util as jtu
from jaxtyping import Array, PRNGKeyArray


def half_precision(model: eqx.Module) -> eqx.Module:
    return jtu.tree_map(lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x, model)

def save_eqx_obj(save_dir: str, filename: str, obj: tuple):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    eqx.tree_serialise_leaves(filename, obj)
        
def load_eqx_obj(filepath: str, obj: tuple) -> tuple:
    return eqx.tree_deserialise_leaves(path_or_file=filepath,
                                       like=obj)

def count_params(model: eqx.Module):
    params_fn = lambda model: sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    num_params, non_embed_params = params_fn(model), params_fn(model.main_block)
    
    num_params /= 1_000_000
    non_embed_params /= 1_000_000
    
    print(f"\nModel # of parameters: {num_params:.2f}M\n# of recurrent parameters: {non_embed_params:.2f}M\n")

def process_seq(seq):
    return [list(map(jnp.array, subseq)) for subseq in seq]

def convert_to_jax(x: torch.Tensor) -> Array:
    
    if isinstance(x, torch.Tensor):
        return jnp.array(x.detach().cpu().numpy())
    elif isinstance(x, list) and isinstance(x[0], tuple):
        # i.e, x is a list of tuples
        output = process_seq(x)
        output_x, output_y, output_z = zip(*output)
        
        return [jnp.array(i) for i in [output_x, output_y, output_z]]
    else:
        return jnp.array(x)

def get_rand_nums(key: PRNGKeyArray, lower_bound: int, upper_bound: int, bsz: int) -> Array:
    random_numbers = jax.random.randint(key, shape=(bsz,), minval=lower_bound, maxval=upper_bound)
    return random_numbers


def inverted_pyramid(arr: Array, max_iters: int):
    arr_min, arr_max = arr.min(), arr.max()
    mid = (arr_min + arr_max) / 2
    base = jnp.arange(1, max_iters)
    weights = jnp.abs(base - mid)
    
    return jnp.clip(weights, 1)

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    print(get_rand_nums(key, 0, 2, 10))