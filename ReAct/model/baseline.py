from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree
from jmp import Policy

from .blocks import AttentionBlock, LinearProj

policy = Policy(compute_dtype=jnp.bfloat16, param_dtype=jnp.float32, output_dtype=jnp.bfloat16)

# ruff: noqa: E402, E731

class VanillaModule(eqx.Module):
    '''
    Main block of the GPT model where you just compute
    all the attention blocks sequentially
    '''

    attention_blocks: PyTree[AttentionBlock]
    
    def __init__(self,
                 seqlen: int,
                 bottleneck: int,
                 n_heads: int,
                 drop_rate: float,
                 num_blocks: int,
                 key: PRNGKeyArray):
        
        keys = jax.random.split(key, num_blocks)
        make_block: Callable = lambda k: AttentionBlock(seqlen, n_heads, drop_rate, bottleneck, k)  # noqa: E731

        self.attention_blocks = [
            eqx.filter(make_block(k), eqx.is_array_like) for k in keys
        ]

    def __call__(
        self,
        input_arr: Array,
        pad_mask: Array,
        enable_dropout: bool = True,
        key: Optional[PRNGKeyArray] = None,
    ) -> Array:
        
        input_arr, pad_mask = policy.cast_to_compute((input_arr, pad_mask))
        
        def f(input_arr: Array, block: PyTree):
            input_arr = policy.cast_to_compute(input_arr)

            output = block(input_arr, input_arr, pad_mask, enable_dropout, key)

            return policy.cast_to_output(output), None

        out = input_arr

        for block in self.attention_blocks:
            out, _ = f(out, block)
        
        return policy.cast_to_output(out)
        
class GPT(eqx.Module):
    '''
    Vanilla Transformer model
    '''
    __name__ = 'GPT'
    
    embed_layer: eqx.nn.Embedding
    embed_ln: eqx.nn.LayerNorm
    unemb_ln: eqx.nn.LayerNorm
    main_block: VanillaModule
    out_head: LinearProj
    
    def __init__(self,
                 n_heads: int,
                 seqlen: int,
                 num_blocks: int,
                 width: int,
                 drop_rate: float,
                 vocab_size: int,
                 key: PRNGKeyArray):
        
        keys = jax.random.split(key, 3)
        
        # Custom initialization for the Embedding Layer
        embed_weights: Array = jax.random.normal(
            key, (vocab_size, width), dtype=jnp.float32
        ) * ((2 / (5 * width)) ** 0.5)

        self.embed_ln = eqx.nn.LayerNorm(width)
        self.unemb_ln = eqx.nn.LayerNorm(width)
        self.embed_layer = eqx.nn.Embedding(weight=embed_weights)

        self.main_block = VanillaModule(seqlen, width, n_heads, drop_rate, num_blocks, key=keys[1])
        self.out_head = LinearProj(width, vocab_size, key=keys[2])
    
    @eqx.filter_jit
    def __call__(self,
                 input_arr: Array,
                 pad_mask: Array,
                 enable_dropout: bool,
                 key: PRNGKeyArray) -> Array:

        embed_fn = lambda x: self.embed_ln(self.embed_layer(x))

        input_arr = jax.vmap(embed_fn)(input_arr) # (batch, seqlen, bottleneck)

        input_arr, pad_mask = policy.cast_to_compute((input_arr, pad_mask))

        output = self.main_block(input_arr, pad_mask, enable_dropout, key)

        output = jax.vmap(self.unemb_ln)(output)

        return self.out_head(output)
