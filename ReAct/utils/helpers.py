import os
from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import torch
from jaxtyping import Array, PRNGKeyArray

def save_eqx_obj(save_dir: str, filename: str, obj: tuple):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    eqx.tree_serialise_leaves(filename, obj)
        
def load_eqx_obj(filepath: str, obj: tuple) -> tuple:
    return eqx.tree_deserialise_leaves(path_or_file=filepath,
                                       like=obj)

def count_params(model: eqx.Module):
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    num_millions = num_params / 1_000_000
    print(f"Model # of parameters: {num_millions:.2f}M")

def process_seq(seq):
    return [list(map(jnp.array, subseq)) for subseq in seq]

def convert_to_jax(x: torch.Tensor) -> Array:
    if isinstance(x, torch.Tensor):
        return jnp.array(x.detach().cpu().numpy())
    elif isinstance(x, list):
        # x is a list of tuples
        output = process_seq(x)
        output_x, output_y, output_z = zip(*output)
        
        return jnp.stack(output_x), jnp.stack(output_y), jnp.stack(output_z)
    else:
        return jnp.array(x)

def get_rand_nums(key: PRNGKeyArray, lower_bound: int, upper_bound: int, bsz: int) -> Array:
    random_numbers = jax.random.randint(key, shape=(bsz,), minval=lower_bound, maxval=upper_bound)
    return random_numbers

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    print(get_rand_nums(key, 0, 2, 10))