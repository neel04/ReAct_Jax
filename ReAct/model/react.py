from typing import Any, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn import LayerNorm
from jaxtyping import Array, PRNGKeyArray, PyTree

from ReAct.utils.helpers import megatron_init
from ReAct.utils.sharding import Sharding

from .blocks import (
    AdaptableAttentionBlock,
    FastEmbedding,
    LinearProj,
    UnsharedBlock,
)

# ruff: noqa: E402, E731

class RecurrentModule(eqx.Module):
    '''
    Bunch of Attentionlayers in a pseuo-LSTM fashion
    '''
    sharding: Sharding = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    max_iters: int = eqx.field(static=True)

    reshape_gate: LinearProj
    attention_layers: List[PyTree]
    post_ln: eqx.nn.LayerNorm
    unshared_layers: UnsharedBlock[LinearProj | LayerNorm]

    def __init__(
        self,
        rank: int,
        seqlen: int,
        drop_rate: float,
        n_heads: int,
        num_layers: int,
        bottleneck: int,
        max_iters: int,
        key: PRNGKeyArray,
        strategy: Sharding
    ):

        self.sharding = strategy
        self.num_layers = num_layers
        self.max_iters = max_iters

        keys = jax.random.split(key, num_layers)

        make_attn = lambda k: self.make_layer(
            self.sharding,
            rank=rank,
            seqlen=seqlen,
            n_heads=n_heads,
            num_layers=num_layers,
            drop_rate=drop_rate,
            bottleneck=bottleneck,
            max_iters=max_iters,
            key=k,
        )

        self.post_ln = eqx.nn.LayerNorm(bottleneck)

        self.reshape_gate = LinearProj(
            bottleneck * 2, bottleneck, strategy=self.sharding, key=key
        )

        self.unshared_layers = UnsharedBlock(
            layers={
                "post_ln": LayerNorm(bottleneck),
            },
            num_repeats=max_iters,
            key=key,
        )

        self.attention_layers = [make_attn(k) for k in keys] # disable `scan`-over layers for now

    @staticmethod
    def make_layer(
        strategy: Sharding,
        rank: int,
        seqlen: int,
        n_heads: int,
        num_layers: int,
        drop_rate: float,
        bottleneck: int,
        max_iters: int,
        key: PRNGKeyArray,
    ) -> AdaptableAttentionBlock:
        return AdaptableAttentionBlock(
            rank,
            seqlen,
            n_heads,
            num_layers,
            drop_rate,
            bottleneck,
            max_iters=max_iters,
            key=key,
            strategy=strategy,
        )

    def __call__(
        self,
        prev_latent: Array,
        post_iters: Tuple[Array, Array],
        input_arr: Array,
        pad_mask: Array,
        enable_dropout: bool,
        iteration_index: int,
        key: PRNGKeyArray,
    ) -> Array:

        keys = jax.random.split(key, self.num_layers * self.max_iters)

        x = jnp.concatenate([prev_latent, input_arr], axis=-1)

        x = self.reshape_gate(x)

        x, pad_mask = self.sharding.cast((x, pad_mask))

        def scan_fn(
            carry: Tuple[Array, int], layer: AdaptableAttentionBlock
        ) -> Tuple[Tuple[Array, int], Array]:
            x, idx = carry

            blck_global_idx = idx + (self.max_iters * iteration_index)

            x = layer(
                x,
                post_iters,
                pad_mask,
                enable_dropout,
                keys[blck_global_idx],
            )

            x = self.unshared_layers.apply_layer(
                "post_ln", iteration_index, (x,), eqx.filter_vmap
            )

            x = self.sharding.cast(x)

            return (x, idx + 1), x

        carry = (x, 0)

        for layer in self.attention_layers:
            carry, _ = scan_fn(carry, layer)

        return self.sharding.cast(carry[0])


class React(eqx.Module):
    '''
    The core ReAct model that holds utilities for performing recursive iterations
    '''
    __name__ = 'ReAct'

    sharding: Sharding = eqx.field(static=True)
    max_iters: int = eqx.field(static=True)
    seqlen: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    
    embed_layer: FastEmbedding
    embed_ln: eqx.nn.LayerNorm
    main_block: RecurrentModule
    unshared_layers: UnsharedBlock[LayerNorm]
    unemb_ln: eqx.nn.LayerNorm
    out_head: LinearProj
    post_attn_head: LinearProj
    post_mlp_head: LinearProj

    def __init__(
        self,
        rank: int,
        n_heads: int,
        seqlen: int,
        max_iters: int,
        num_blocks: int,
        width: int,
        drop_rate: float,
        vocab_size: int,
        key: PRNGKeyArray,
        strategy: Any
    ):
        key1, key2, key3 = jax.random.split(key, 3)

        self.sharding = strategy
        self.max_iters = max_iters
        self.width = width
        self.seqlen = seqlen

        self.embed_ln = eqx.nn.LayerNorm(width)
        self.embed_layer = FastEmbedding(vocab_size, width, key1, strategy)

        self.main_block = RecurrentModule(
            rank,
            seqlen,
            drop_rate,
            n_heads,
            num_blocks,
            width,
            max_iters,
            key2,
            self.sharding,
        )

        self.unshared_layers = UnsharedBlock(
            layers={
                "post_ln": LayerNorm(width),
            },
            num_repeats=max_iters,
            key=key,
        )

        self.post_attn_head = LinearProj(seqlen, width, key1, strategy=self.sharding)
        self.post_mlp_head = LinearProj(seqlen, width, key2, strategy=self.sharding)

        self.unemb_ln = eqx.nn.LayerNorm(width)
        self.out_head = LinearProj(width, vocab_size, key=key3, strategy=self.sharding)

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
        
        keys = jax.random.split(key, iters_to_do)

        post_iters = (
            megatron_init(dims=(self.width, self.width), key=keys[0]),
            megatron_init(dims=(self.width, self.width), key=keys[1]),
        )

        interim_thought, input_arr, mask = self.sharding.cast((interim_thought, input_arr, mask))

        def body_fun(
            carry: Tuple[Array, Tuple[Array, Array]], idx: int
        ) -> Tuple[Tuple[Array, Tuple[Array, Array]], Array]:
            input, post_iters = carry

            latent = self.main_block(
                input,
                post_iters,
                input_arr,
                mask,
                enable_dropout,
                idx,
                keys[idx],
            )  # (seqlen, width)

            post_iters = (self.post_attn_head(latent.T), self.post_mlp_head(latent.T))

            latent = self.unshared_layers.apply_layer(
                "post_ln", idx, args=(latent,), modifier_fn=eqx.filter_vmap
            )

            latent = self.sharding.cast(latent)

            return (latent, post_iters), latent

        output, _ = eqx.internal.scan(
            f=body_fun,
            init=(interim_thought, post_iters),
            xs=jnp.arange(iters_to_do), # type: ignore
            kind="checkpointed",
            checkpoints=iters_to_do,
        )

        return output[0]

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

        input_arr, interim_thought = self.sharding.cast((input_arr, interim_thought))

        output = self.iterate_for_steps(
            interim_thought, input_arr, pad_mask, iters_to_do, is_training, key
        )  # (batch, seqlen, bottleneck)

        output = jax.vmap(self.unemb_ln)(output)

        return self.out_head(output), output
