from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree
from jmp import Policy

from ReAct.utils.sharding import Sharding

from .blocks import AttentionBlock, LinearProj

policy = Policy(compute_dtype=jnp.bfloat16, param_dtype=jnp.float32, output_dtype=jnp.bfloat16)

# ruff: noqa: E402, E731

class VanillaModule(eqx.Module):
    '''
    Main block of the GPT model where you just compute
    all the attention blocks sequentially
    '''

    sharding: Sharding = eqx.field(static=True)
    attention_blocks: PyTree[AttentionBlock]
    
    def __init__(
        self,
        seqlen: int,
        bottleneck: int,
        n_heads: int,
        drop_rate: float,
        num_blocks: int,
        key: PRNGKeyArray,
        strategy: Sharding
    ):
        
        self.sharding = strategy
        keys = jax.random.split(key, num_blocks)

        make_attn = lambda k: self.make_layer(
            self.sharding, seqlen, n_heads, drop_rate, bottleneck, k
        )

        self.attention_blocks = eqx.filter(
            eqx.filter_vmap(make_attn)(keys), eqx.is_array_like
        )

    @staticmethod
    def make_layer(strategy: Sharding, seqlen: int, n_heads: int, drop_rate: float, bottleneck: int, key: PRNGKeyArray) -> AttentionBlock:
        return AttentionBlock(seqlen, n_heads, drop_rate, bottleneck, key, strategy)

    def __call__(
        self,
        input_arr: Array,
        pad_mask: Array,
        enable_dropout: bool = True,
        key: Optional[PRNGKeyArray] = None,
    ) -> Array:
        
        input_arr, pad_mask = self.sharding.shard_cast((input_arr, pad_mask))

        dynamic_part, static_part = eqx.partition(
            self.attention_blocks,
            eqx.is_array_like,
            is_leaf=lambda x: isinstance(x, eqx.nn.Dropout),
        )

        def scan_f(carry: Array, _dy_blck: PyTree):
            _input_arr = self.sharding.shard_cast(carry)

            block = eqx.combine(_dy_blck, static_part)
            output = block(_input_arr, _input_arr, pad_mask, enable_dropout, key)

            return self.sharding.shard_cast(output), None

        final_output, _ = eqx.internal.scan(
            f=scan_f,
            init=input_arr,
            xs=dynamic_part,
            kind="checkpointed"
        )

        return policy.cast_to_output(final_output)
        
class GPT(eqx.Module):
    """
    Vanilla Transformer model
    """

    __name__ = "GPT"

    sharding: Sharding = eqx.field(static=True)
    embed_layer: eqx.nn.Embedding
    embed_ln: eqx.nn.LayerNorm
    unemb_ln: eqx.nn.LayerNorm
    main_block: VanillaModule
    out_head: LinearProj

    def __init__(
        self,
        n_heads: int,
        seqlen: int,
        num_blocks: int,
        width: int,
        drop_rate: float,
        vocab_size: int,
        key: PRNGKeyArray,
        strategy: Any,
    ):
        self.sharding = strategy
        keys = jax.random.split(key, 3)

        # Custom initialization for the Embedding Layer
        embed_weights: Array = jax.random.normal(
            key, (vocab_size, width), dtype=jnp.float32
        ) * ((2 / (5 * width)) ** 0.5)

        self.embed_ln = eqx.nn.LayerNorm(width)
        self.unemb_ln = eqx.nn.LayerNorm(width)
        self.embed_layer = eqx.nn.Embedding(weight=embed_weights)

        self.main_block = VanillaModule(
            seqlen,
            width,
            n_heads,
            drop_rate,
            num_blocks,
            key=keys[1],
            strategy=self.sharding,
        )

        self.out_head = LinearProj(
            width, vocab_size, key=keys[2], strategy=self.sharding
        )

    @eqx.filter_jit
    def __call__(
        self, input_arr: Array, pad_mask: Array, enable_dropout: bool, key: PRNGKeyArray
    ) -> Array:

        input_arr = self.sharding.shard_cast(input_arr)
        embed_fn = lambda x: self.embed_ln(self.embed_layer(x))

        input_arr = jax.vmap(embed_fn)(input_arr)  # (batch, seqlen, bottleneck)
        input_arr, pad_mask = self.sharding.shard_cast((input_arr, pad_mask))

        output = self.main_block(input_arr, pad_mask, enable_dropout, key)
        output = self.sharding.shard_cast(output)
        output = jax.vmap(self.unemb_ln)(output)

        return self.out_head(output)
