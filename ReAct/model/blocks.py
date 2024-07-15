import math
from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, Int
from jmp import Policy

policy = Policy(compute_dtype=jnp.bfloat16, param_dtype=jnp.float32, output_dtype=jnp.bfloat16)

# ruff: noqa: F722

class AttentionBlock(eqx.Module):
    """Basic Block for LiteAttention"""

    seqlen: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    in_dim: int = eqx.field(static=True)

    attn_gate: eqx.Module
    rope_embed: eqx.Module
    ln1: eqx.Module
    ln2: eqx.Module
    mlp: eqx.Module

    def __init__(
        self,
        seqlen: int,
        n_heads: int,
        drop_rate: float,
        in_dim: int,
        key: PRNGKeyArray = jax.random.PRNGKey(69),
    ):
        key1, key2 = jax.random.split(key, 2)

        self.seqlen = seqlen
        self.n_heads = n_heads
        self.in_dim = in_dim

        self.rope_embed = eqx.nn.RotaryPositionalEmbedding(embedding_size=in_dim // n_heads)

        self.attn_gate = eqx.nn.MultiheadAttention(num_heads=n_heads, query_size=in_dim,
                                                   use_query_bias=True, use_key_bias=True,
                                                   use_value_bias=True, use_output_bias=True, 
                                                   dropout_p=drop_rate, key=key1)

        self.ln1 = eqx.nn.LayerNorm(self.in_dim)
        self.ln2 = eqx.nn.LayerNorm(self.in_dim)

        self.mlp = MLP(self.in_dim, self.in_dim, drop_rate, key2)

    def process_heads(
        self,
        query_heads: Float[Array, "seq_length num_heads qk_size"],
        key_heads: Float[Array, "seq_length num_heads qk_size"],
        value_heads: Float[Array, "seq_length num_heads vo_size"],
    ) -> tuple[
        Float[Array, "seq_length num_heads qk_size"],
        Float[Array, "seq_length num_heads qk_size"],
        Float[Array, "seq_length num_heads vo_size"],
    ]:
        query_heads = jax.vmap(self.rope_embed, in_axes=1, out_axes=1)(query_heads)

        key_heads = jax.vmap(self.rope_embed, in_axes=1, out_axes=1)(key_heads)

        return query_heads, key_heads, value_heads

    def _make_self_attention_mask(self, pad_mask: Int[Array, "seqlen"]) -> Array:
        mask = jnp.ones((self.seqlen, self.seqlen))
        mask = jnp.tril(mask) * pad_mask
        mask = jnp.expand_dims(mask, 0)

        return jnp.repeat(mask, self.n_heads, axis=0)

    def __call__(
        self,
        inp: Float[Array, "seqlen in_dim"],
        input_arr: Array,
        mask: Array,
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, "seqlen in_dim"]:

        key_1, key_2 = jax.random.split(key, 2)
        inp, input_arr, mask = policy.cast_to_compute((inp, input_arr, mask))
        
        x = jax.vmap(self.ln1)(inp)

        inp += self.attn_gate(x, input_arr, input_arr,
                              mask=self._make_self_attention_mask(mask),
                              inference=enable_dropout,
                              process_heads=self.process_heads, key=key_1)

        x = jax.vmap(self.ln2)(inp)

        inp += self.mlp(x, enable_dropout=True, key=key_2)

        return policy.cast_to_output(inp)
    
    
class NewGELU(eqx.Module):
    def __call__(self, x: jax.Array, *args) -> jax.Array:
        c: float = math.sqrt(2.0 / math.pi)
        a: float = 0.044715

        x = policy.cast_to_compute(x)
        output =  0.5 * x * (1.0 + jax.nn.tanh(c * (x + a * jnp.power(x, 3.0))))
        
        return policy.cast_to_output(output)

class MLP(eqx.Module):
    '''A simple MLP - w/ Dropout'''

    layer_1: eqx.Module
    layer_2: eqx.Module
    dropout: eqx.nn.Dropout
    act: callable

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 p: float,
                 key: PRNGKeyArray):
        
        key1, key2 = jax.random.split(key, 2)

        self.layer_1 = LinearProj(input_dim, output_dim * 4, key=key1)
        self.layer_2 = LinearProj(output_dim * 4, output_dim, key=key2)
        self.act = NewGELU()

        self.dropout = eqx.nn.Dropout(p=p)

    def __call__(self, x: Array, enable_dropout: bool, key: PRNGKeyArray):
        x = policy.cast_to_compute(x)
        
        x = self.act(self.layer_1(x))
        x = self.layer_2(x)
        
        output = self.dropout(x, key=key, inference=enable_dropout)

        return policy.cast_to_output(self.act(output))

class lerp(eqx.Module):
    alpha: Array

    def __init__(self, alpha: float = 0.5):
        self.alpha = jnp.array([alpha])

    def __call__(self, x: Array, y: Array) -> Array:
        x, y = policy.cast_to_compute((x, y))

        output = self.alpha * x + (1 - self.alpha) * y

        return policy.cast_to_output(output)

class GatedBlock(eqx.Module):
    """Gated Block for any general function"""
    
    gate: eqx.Module
    block: eqx.Module
    ln: eqx.Module
    activation: Callable
    
    def __init__(
        self,
        fun: Callable,
        args: Tuple,
        in_dim: int,
        key: PRNGKeyArray,
    ):

        self.block = fun(*args) if args else fun

        self.gate = LinearProj(in_dim, in_dim, key=key)
        self.ln = eqx.nn.LayerNorm(in_dim)
        self.activation = NewGELU()
    
    def __call__(self, x: Array, ctx: Array, call_args: Tuple) -> Array:
        x, ctx = policy.cast_to_compute((x, ctx))

        x = self.block(x, ctx, *call_args)
        x = jax.vmap(self.ln)(x)

        x *= jax.nn.silu(self.gate(ctx))
        
        return policy.cast_to_output(x)

class GatedMLP(eqx.Module):
    '''
    Gated MLP, Mamba-ish style
    '''

    ln: eqx.Module
    up_proj: eqx.Module
    down_proj: eqx.Module
    gate: eqx.Module
    activation: callable

    def __init__(self, input_dim: int, output_dim: int, key: PRNGKeyArray):
        key_1, key_2, key_3, key_4 = jax.random.split(key, 4)

        self.up_proj = LinearProj(input_dim, output_dim, key=key_1)
        self.gate = LinearProj(input_dim, output_dim, key=key_4)
        self.down_proj = LinearProj(output_dim, output_dim, key=key_3)

        self.ln = eqx.nn.LayerNorm(output_dim)

        self.activation = NewGELU()

    def __call__(self, arr: Array, cond: Optional[Array] = None) -> Array:
        cond = arr if cond is None else cond
        x, cond = policy.cast_to_compute((arr, cond))
        
        x = self.activation(self.up_proj(arr))
        x = jax.vmap(self.ln)(x)
        x = self.down_proj(x * jax.nn.silu(self.gate(cond)))
        
        return policy.cast_to_output(x)

class DynamicGatedMLP(eqx.Module):
    proj_1: eqx.Module
    proj_2: eqx.Module
    ln: eqx.Module
    act: callable

    def __init__(self, input_dim: int, key: PRNGKeyArray) -> Array:

        keys = jax.random.split(key, 2)
        output_dim = input_dim * 2

        self.proj_1 = LinearProj(input_dim, output_dim, key=keys[0])
        self.proj_2 = LinearProj(output_dim, output_dim, key=keys[1])
        self.ln = eqx.nn.LayerNorm(output_dim)
        self.act = NewGELU()

    def __call__(
        self, input_arr: Float[Array, "seqlen width"]
    ) -> Float[Array, "seqlen width"]:

        input_arr = policy.cast_to_param(input_arr)

        dynamic_weights = self.proj_2(self.act(self.proj_1(input_arr)))
        dynamic_weights = policy.cast_to_param(dynamic_weights)
        dynamic_weights = jax.vmap(self.ln)(dynamic_weights)
        dw_1, dw_2 = jnp.split(dynamic_weights, 2, axis=-1)
        output = (input_arr @ dw_2.T) @ dw_1

        return policy.cast_to_output(output)

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
        self.weight = jax.random.uniform(wkey, (input_dim, output_dim), minval=-lim, maxval=lim)

        if use_bias:
            self.bias = jax.random.uniform(bkey, (output_dim,), minval=-lim, maxval=lim)
        else:
            self.bias = jnp.zeros((output_dim,))
    
    def __call__(self,
                 arr: Float[Array, 'batch in_dim'],
                 mask: Optional[Array] = None,
                 **kwargs) -> Array:
        
        arr, mask = policy.cast_to_compute((arr, mask))
        
        mask = jnp.ones_like(self.weight) if mask is None else mask
        output = arr @ (self.weight * mask.astype(arr.dtype)) + self.bias
        
        return policy.cast_to_output(output)

class LiteAttention(eqx.Module):
    input_dim: int = eqx.field(static=True)
    weight: eqx.Module

    def __init__(self, input_dim: int, key: PRNGKeyArray):
        self.input_dim = input_dim
        self.weight = LinearProj(input_dim, input_dim, use_bias=True, key=key)

    @jax.jit
    def __call__(self, x: Float[Array, 'seqlen in_dim'], mask: Array):
        mask, x = policy.cast_to_compute((mask, x))
        attn_weights = jax.nn.softmax(self.weight(x.T, mask), axis=1) # type: ignore
        output = x * attn_weights.T
        return policy.cast_to_output(output)

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
  
    def __call__(self, x: Float[Array, 'seqlen in_dim'], mask: Array, enable_dropout: bool, key: PRNGKeyArray) -> Array:
        x, mask = policy.cast_to_compute((x, mask))
        
        arr = x.T
        arr = self.act_1(self.token_mixer(arr, key=key, mask=mask))
        arr = jax.vmap(self.norm)(arr.T)
        x = x + arr
        output = x + self.act_2(self.channel_mixer(arr, enable_dropout, key=key))
        
        return policy.cast_to_output(output)