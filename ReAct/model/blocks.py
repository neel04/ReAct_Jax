import math
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, BFloat16, PRNGKeyArray

# ruff: noqa: F722
class AttentionBlock(eqx.Module):
    """Basic Block for LiteAttention"""

    seqlen: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    bottleneck: int = eqx.field(static=True)
    
    attn_gate: eqx.Module
    ln1: eqx.Module
    ln2: eqx.Module
    mlp: eqx.Module

    def __init__(self,
                 seqlen: int,
                 n_heads: int,
                 drop_rate: float,
                 bottleneck: int,
                 key: PRNGKeyArray):
        
        key1, key2 = jax.random.split(key, 2)

        self.seqlen = seqlen
        self.n_heads = n_heads
        self.bottleneck = bottleneck

        self.attn_gate = eqx.nn.MultiheadAttention(num_heads=n_heads, query_size=bottleneck,
                                                   use_query_bias=True, use_key_bias=True,
                                                   use_value_bias=True, use_output_bias=True, 
                                                   dropout_p=drop_rate, key=key1)

        self.ln1 = eqx.nn.LayerNorm(self.bottleneck)
        self.ln2 = eqx.nn.LayerNorm(self.bottleneck)

        self.mlp = MLP(self.bottleneck, self.bottleneck, drop_rate, key2)

    def _make_self_attention_mask(self,
                                  pad_mask: Array) -> Array:
        
        """Create self-attention mask from sequence-level mask."""
        
        mask = jnp.ones((self.seqlen, self.seqlen), dtype=jnp.bfloat16)
        mask = jnp.tril(mask)
        mask = jnp.expand_dims(mask, 0)
        return jnp.repeat(mask, self.n_heads, axis=0)
    
    def _make_mixer_mask(self,
                         pad_mask: Array):
        
        # Almost same, but we triu instead of tril
        # and we don't need to merge with pad_mask
        mask = jnp.ones((self.seqlen, self.seqlen)) * pad_mask
        mask = jnp.triu(mask)
        
        return mask

    def __call__(self,
                 inp: BFloat16[Array, 'seqlen bottleneck'],
                 input_arr: Array,
                 mask: Array,
                 enable_dropout: bool,
                 key: PRNGKeyArray):
        
        key_1, key_2 = jax.random.split(key, 2)
        inp = inp.astype(jnp.bfloat16)
        
        x = jax.vmap(self.ln1)(inp)
        inp += self.attn_gate(x, input_arr, input_arr,
                            mask=self._make_self_attention_mask(mask),
                            key=key_1, inference=enable_dropout)
        
        x = jax.vmap(self.ln2)(inp)
        inp += self.mlp(x, enable_dropout=True, key=key_2)

        return inp.astype(jnp.bfloat16)
    
    
class NewGELU(eqx.Module):
    def __call__(self, x: jax.Array, *args) -> jax.Array:
        c: float = math.sqrt(2.0 / math.pi)
        a: float = 0.044715
        return 0.5 * x * (1.0 + jax.nn.tanh(c * (x + a * jnp.power(x, 3.0))))

class MLP(eqx.Module):
    '''A simple MLP - w/ Dropout'''
    layers: eqx.nn.Sequential
    dropout: eqx.nn.Dropout

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 p: float,
                 key: PRNGKeyArray):
        
        key1, key2 = jax.random.split(key, 2)

        self.layers = [
            LinearProj(input_dim, output_dim * 4, key=key1),
            eqx.nn.Lambda(NewGELU()),
            LinearProj(output_dim * 4, output_dim, key=key2),
            ]

        self.dropout = eqx.nn.Dropout(p=p)

    def __call__(self,
                 x: Array,
                 enable_dropout: bool,
                 key: PRNGKeyArray):
        
        x = x.astype(jnp.bfloat16)
        
        for layer in self.layers:
            x = layer(x).astype(jnp.bfloat16)
        
        return self.dropout(x, key=key, inference=enable_dropout)

class LinearProj(eqx.Module):
    bias: Optional[jax.Array]
    weight: jax.Array

    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 key: PRNGKeyArray,
                 use_bias=True):
        
        assert input_dim >= 1 or output_dim >= 1, f'input_dim: {input_dim} | output_dim: {output_dim} are too small'
        wkey, bkey = jax.random.split(key, 2)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        lim = 1 / math.sqrt(input_dim)
        self.weight = jax.random.uniform(wkey, (input_dim, output_dim), minval=-lim, maxval=lim).astype(jnp.bfloat16)

        if use_bias:
            self.bias = jax.random.uniform(bkey, (output_dim,), minval=-lim, maxval=lim).astype(jnp.bfloat16)
        else:
            self.bias = jnp.zeros((output_dim,)).astype(jnp.bfloat16)
    
    def __call__(self,
                 input: BFloat16[Array, 'batch in_dim'],
                 **kwargs) -> Array:
        
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