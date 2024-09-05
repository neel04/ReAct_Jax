from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree
from jmp import Policy

from .blocks import AttentionBlock, LinearProj

policy = Policy(compute_dtype=jnp.bfloat16, param_dtype=jnp.float32, output_dtype=jnp.bfloat16)

# ruff: noqa: E402, E731

class RecurrentModule(eqx.Module):
    '''
    Bunch of Attentionlayers in a pseuo-LSTM fashion
    '''
    num_layers: int = eqx.field(static=True)

    attention_layers: Array
    post_ln: eqx.nn.LayerNorm
    reshape_gate: LinearProj

    def __init__(
        self,
        seqlen: int,
        drop_rate: float,
        n_heads: int,
        num_layers: int,
        bottleneck: int,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, num_layers)

        make_attn = lambda k: self.make_layer(
            seqlen, n_heads, drop_rate, bottleneck, k
        )

        self.num_layers = num_layers
        self.post_ln = eqx.nn.LayerNorm(bottleneck)
        self.reshape_gate = LinearProj(bottleneck * 2, bottleneck, key=keys[0])

        self.attention_layers = eqx.filter(eqx.filter_vmap(make_attn)(keys), eqx.is_array_like)

    @staticmethod
    def make_layer(seqlen: int, n_heads: int, drop_rate: float, bottleneck: int, key: PRNGKeyArray) -> AttentionBlock:
        return AttentionBlock(seqlen, n_heads, drop_rate, bottleneck, key)

    def __call__(
        self,
        x: Array,
        pad_mask: Array,
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Array:

        keys = jax.random.split(key, self.num_layers)

        dynamic_part, static_part = eqx.partition(
            self.attention_layers,
            eqx.is_array_like,
            is_leaf=lambda x: isinstance(x, eqx.nn.Dropout),
        )

        x, pad_mask = policy.cast_to_compute((x, pad_mask))  # (seqlen, bottleneck)

        x = self.reshape_gate(x) # downsample the concatenated array
        
        def f(input_tup: Tuple[Array, int], _dynamic_bl: PyTree) -> Tuple[Tuple[Array, int], Array]:
            x, idx = input_tup
            layer = eqx.combine(_dynamic_bl, static_part) # reconstruct the layer

            x = layer(x, x, pad_mask, enable_dropout, keys[idx])

            x = jax.vmap(self.post_ln)(x)

            x = policy.cast_to_compute(x) # casting to bf16
            
            return (x, idx + 1), x

        out, _ = jax.lax.scan(f=f, init=(x, 0), xs=dynamic_part, unroll=True)

        return policy.cast_to_output(out[0])

class React(eqx.Module):
    '''
    The core ReAct model that holds utilities for performing recursive iterations
    '''
    __name__ = 'ReAct'

    max_iters: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    
    embed_layer: eqx.nn.Embedding
    embed_ln: eqx.nn.LayerNorm
    main_block: RecurrentModule
    post_ln: eqx.nn.LayerNorm
    unemb_ln: eqx.nn.LayerNorm
    out_head: LinearProj

    def __init__(
        self,
        n_heads: int,
        seqlen: int,
        max_iters: int,
        num_blocks: int,
        width: int,
        drop_rate: float,
        vocab_size: int,
        key: PRNGKeyArray,
    ):
        key1, key2, key3 = jax.random.split(key, 3)

        self.max_iters = max_iters
        self.width = width

        # Custom initialization for the Embedding Layer
        embed_weights: Array = jax.random.normal(
            key1, (vocab_size, width), dtype=jnp.float32
        ) * ((2 / (5 * width)) ** 0.5)

        self.embed_ln = eqx.nn.LayerNorm(width)
        self.embed_layer = eqx.nn.Embedding(weight=embed_weights)
        self.main_block = RecurrentModule(seqlen, drop_rate, n_heads, num_blocks, width, key=key2)

        self.post_ln = eqx.nn.LayerNorm(width)
        self.unemb_ln = eqx.nn.LayerNorm(width)
        self.out_head = LinearProj(width, vocab_size, key=key3)

    @eqx.filter_jit
    def iterate_for_steps(
        self,
        interim_thought: Array,
        input_arr: Array,
        mask: Array,
        iters_to_do: int,
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Array:
        
        input_arr, interim_thought, mask = policy.cast_to_compute((input_arr, interim_thought, mask))

        def body_fun(latent: Array, idx: int) -> Tuple[Array, Array]:
            latent = jnp.concatenate([input_arr, latent], axis=-1) 

            latent = self.main_block(latent, mask, enable_dropout, key)  # (seqlen, width)

            latent = jax.vmap(self.post_ln)(latent)  # Post-LN for stability

            latent = policy.cast_to_output(latent)
            
            return latent, latent

        output, _ = eqx.internal.scan(
            f=body_fun,
            init=interim_thought,
            xs=jnp.arange(iters_to_do),
            kind="checkpointed",
            checkpoints=iters_to_do,
        )

        return output

    @eqx.filter_jit
    def __call__(
        self,
        input_arr: Array | Tuple[Array, Array],
        iters_to_do: int,
        pad_mask: Array,
        prev_thought: bool = False,
        is_training: bool = True,
        key: PRNGKeyArray = jax.random.PRNGKey(0),
    ) -> Tuple[Array, Array]:

        embed_fn = lambda x: self.embed_ln(self.embed_layer(x))

        if prev_thought:
            assert isinstance(input_arr, tuple), 'prev_thought is True, but input_arr is not a tuple'
            input_arr, interim_thought = input_arr
            input_arr = jax.vmap(embed_fn)(input_arr) # (batch, seqlen, bottleneck)
        else:
            input_arr = jax.vmap(embed_fn)(input_arr)  # (batch, seqlen, bottleneck)
            interim_thought = input_arr.copy()  # has to be a copy of the embedded + normed array

        input_arr, interim_thought = policy.cast_to_compute((input_arr, interim_thought))

        output = self.iterate_for_steps(interim_thought, input_arr, pad_mask, iters_to_do, is_training, key)  # (batch, seqlen, bottleneck)

        output = jax.vmap(self.unemb_ln)(output)

        return self.out_head(output), output
