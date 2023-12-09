import math
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, BFloat16, PRNGKeyArray

# ruff: noqa: F722

class NewGELU(eqx.Module):
    def __call__(self, x: jax.Array, *args) -> jax.Array:
        c: float = math.sqrt(2.0 / math.pi)
        a: float = 0.044715
        return 0.5 * x * (1.0 + jax.nn.tanh(c * (x + a * jnp.power(x, 3.0))))

class MLP(eqx.Module):
    '''A simple MLP - w/ Dropout'''
    layers: eqx.nn.Sequential
    dropout: eqx.nn.Dropout

    def __init__(self, input_dim: int, output_dim: int, p: float, key: PRNGKeyArray):
        key1, key2 = jax.random.split(key, 2)

        self.layers = [
            LinearProj(input_dim, output_dim * 4, key=key1),
            eqx.nn.Lambda(NewGELU()),
            LinearProj(output_dim * 4, output_dim, key=key2),
            ]

        self.dropout = eqx.nn.Dropout(p=p)

    def __call__(self, x: Array, key: PRNGKeyArray):
        for layer in self.layers:
            x = layer(x).astype(jnp.bfloat16)
        
        return self.dropout(x, key=key, inference=False)

class LinearProj(eqx.Module):
    bias: Optional[jax.Array]
    weight: jax.Array

    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(self, input_dim, output_dim, key: PRNGKeyArray, use_bias=True):
        assert input_dim >= 1 or output_dim >= 1, f'input_dim: {input_dim} | output_dim: {output_dim} are too small'
        wkey, bkey = jax.random.split(key, 2)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        lim = 1 / math.sqrt(input_dim)
        self.weight = jax.random.uniform(wkey, (input_dim, output_dim), minval=-lim, maxval=lim).astype(jnp.bfloat16)

        if use_bias:
            self.bias = jax.random.uniform(bkey, (output_dim,), minval=-lim, maxval=lim)
        else:
            self.bias = jnp.zeros((output_dim,))
    
    def __call__(self, input: BFloat16[Array, 'batch in_dim'], **kwargs):
        mask = kwargs.get('mask', None)
        mask = jnp.ones_like(self.weight) if mask is None else mask
        output = input @ (self.weight * mask.astype(input.dtype)) + self.bias
        return output

class LiteAttention(eqx.Module):
    input_dim: int = eqx.field(static=True)
    weight: eqx.Module

    def __init__(self, input_dim: int, key: PRNGKeyArray):
        self.input_dim = input_dim
        self.weight = LinearProj(input_dim, input_dim, use_bias=True, key=key)

    @jax.jit
    def __call__(self, x: BFloat16[Array, 'seqlen in_dim'], mask: Array):
        attn_weights = jax.nn.softmax(self.weight(x.T, mask), axis=1) # type: ignore
        return x * attn_weights.T

class MixerBlock(eqx.Module):
    '''
    MixerBlock from MLP-Mixer
    Is applied in-place for self-attention
    '''
    act_1: eqx.Module
    act_2: eqx.Module
    norm: eqx.Module
    channel_mixer: eqx.Module
    token_mixer: eqx.Module
    
    def __init__(self, input_dim: int, seqlen: int, drop_rate: float, key: PRNGKeyArray):
        key1, key2 = jax.random.split(key, 2)
        
        self.norm = eqx.nn.LayerNorm(input_dim)
        self.act_1 = NewGELU()
        self.act_2 = NewGELU()
        
        self.channel_mixer = MLP(input_dim, input_dim, drop_rate, key=key1)
        self.token_mixer = LinearProj(seqlen, seqlen, key=key2)
  
    def __call__(self, x: BFloat16[Array, 'seqlen in_dim'], mask: Array, key: PRNGKeyArray):
        arr = x.T
        arr = self.act_1(self.token_mixer(arr, key=key, mask=mask))
        arr = jax.vmap(self.norm)(arr.T)
        x = x + arr
        return x + self.act_2(self.channel_mixer(arr, key))
    
if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    LA = LiteAttention(256, key)
    test = jax.random.normal(key, (128, 256))
    print(LA(test).shape)