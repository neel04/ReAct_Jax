import os

import equinox as eqx
import jax
import jax.numpy as jnp
import torch


from jax import tree_util as jtu
from jaxtyping import Array, PRNGKeyArray
from typing import Optional

    
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

def get_rand_nums(key: PRNGKeyArray, lower_bound: int, upper_bound: int, bsz: int, bias_val: Optional[int] = None) -> Array:
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

if __name__ == '__main__':
    import plotly.express as px
    import pandas as pd
    
    key = jax.random.PRNGKey(0)
    out: Array = get_rand_nums(key, 1, 10, 512, 4)
    elems, counts = jnp.unique(out, return_counts=True)
    df = pd.DataFrame({'elems': elems, 'counts': counts})
    fig = px.bar(df, x='elems', y='counts')
    fig.show()