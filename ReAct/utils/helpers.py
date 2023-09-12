import equinox as eqx
import jax
import jax.numpy as jnp
import torch
from jaxtyping import Array, PRNGKeyArray

def load_eqx_obj(obj, filepath):
    with open(filepath, 'rb') as f:
        print(f"Loading {filepath}")
        return eqx.tree_deserialise_leaves(f, obj)

def count_params(model: eqx.Module):
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    num_millions = num_params / 1_000_000
    print(f"Model # of parameters: {num_millions:.2f}M")

def convert_to_jax(x: torch.Tensor) -> Array:
    if isinstance(x, torch.Tensor):
        return jnp.array(x.detach().cpu().numpy())
    else:
        return jnp.array(x)

def get_rand_nums(key: PRNGKeyArray, lower_bound: int, upper_bound: int, bsz: int) -> Array:
    random_numbers = jax.random.randint(key, shape=(bsz,), minval=lower_bound, maxval=upper_bound)
    return random_numbers

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    print(get_rand_nums(key, 0, 2, 10))