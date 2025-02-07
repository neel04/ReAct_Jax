import math
from typing import Callable, Dict, Generic, Tuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from jmp import Policy

from ReAct.utils.sharding import Sharding, get_strategy

policy = Policy(compute_dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, output_dtype=jnp.bfloat16)

# ruff: noqa: F722

L = TypeVar('L', bound='LinearProj')

class NewGELU(eqx.Module):
    sharding: Sharding = eqx.field(static=True)

    def __init__(self, strategy: Sharding) -> None:
        self.sharding = strategy(policy)

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        c = math.sqrt(2.0 / math.pi)
        a = 0.044715

        x = self.sharding.shard_model_cast(x)
        
        output = 0.5 * x * (1.0 + jax.nn.tanh(c * (x + a * jnp.power(x, 3.0))))

        return self.sharding.shard_model_cast(output)


class LinearProj(eqx.Module):
    bias: jax.Array
    weight: jax.Array

    sharding: Sharding = eqx.field(static=True, converter=get_strategy)
    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        key: PRNGKeyArray,
        use_bias = True,
        strategy: Sharding | None = None
    ):
        assert input_dim >= 1 or output_dim >= 1, (
            f"input_dim: {input_dim} | output_dim: {output_dim} are too small"
        )
        assert strategy is not None, "No strategy provided."

        wkey, bkey = jax.random.split(key, 2)

        self.sharding = strategy(policy)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        lim = 1 / math.sqrt(input_dim)

        self.weight = jax.random.uniform(
            wkey, (input_dim, output_dim), minval=-lim, maxval=lim
        ) * math.sqrt(1 / (3 * input_dim))

        if use_bias:
            self.bias = jax.random.uniform(bkey, (output_dim,), minval=-lim, maxval=lim)
        else:
            self.bias = jnp.zeros((output_dim,))

    @eqx.filter_jit
    def __call__(
        self,
        arr: Float[Array, "batch in_dim"],
        mask: Array | None = None,
    ) -> Array:

        arr, mask = self.sharding.shard_model_cast((arr, mask)) 

        _mask = jnp.ones_like(self.weight) if mask is None else mask

        return self.sharding.shard_model_cast(
            arr @ (self.weight * _mask.astype(arr.dtype)) + self.bias
        )


class MLP(eqx.Module):
    """A simple MLP - w/ Dropout"""

    sharding: Sharding = eqx.field(static=True)
    layer_1: LinearProj
    layer_2: LinearProj
    dropout: eqx.nn.Dropout
    act: NewGELU

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        p: float,
        key: PRNGKeyArray,
        strategy: Sharding,
        ff_mult: int = 4
    ):
        key1, key2 = jax.random.split(key, 2)

        self.sharding = strategy(policy)
        
        self.layer_1 = LinearProj(
            input_dim, output_dim * ff_mult, key=key1, strategy=strategy
        )
        self.layer_2 = LinearProj(
            output_dim * ff_mult, output_dim, key=key2, strategy=strategy
        )
        self.act = NewGELU(strategy)

        self.dropout = eqx.nn.Dropout(p=p)

    @eqx.filter_jit
    def __call__(self, x: Array, enable_dropout: bool, key: PRNGKeyArray):
        x = self.sharding.shard_model_cast(x)

        x = self.act(self.layer_1(x))

        x = self.layer_2(x)

        output = self.dropout(x, key=key, inference=enable_dropout)

        return self.sharding.shard_model_cast(self.act(output))

class CopyGate(eqx.Module):
    """
    Copy Gate adapted from Csordas et al.
    """

    sharding: Sharding = eqx.field(static=True)
    gating_layer: MLP

    def __init__(
        self, d_model: int, drop_rate: float, key: PRNGKeyArray, strategy: Sharding
    ):
        self.sharding = strategy(policy)
        self.gating_layer = MLP(d_model, d_model, drop_rate, key, strategy, ff_mult=1)

        # Set biases to -3 to ensure no update at the start
        self.gating_layer = eqx.tree_at(
            lambda m: m.layer_2.bias,
            self.gating_layer,
            replace_fn=lambda arr: jnp.full(arr.shape, -3.0),
        )

    def __call__(self, input: Array, enable_dropout: bool, key: PRNGKeyArray) -> Array:
        return jax.nn.sigmoid(self.gating_layer(input, enable_dropout, key))

class NDRAttentionBlock(eqx.Module):
    """CopyGated Augmented block inspired by Csordas et al."""

    sharding: Sharding = eqx.field(static=True)
    seqlen: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    in_dim: int = eqx.field(static=True)

    attn_gate: eqx.nn.MultiheadAttention
    rope_embed: eqx.nn.RotaryPositionalEmbedding
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    copy_gate: CopyGate
    mlp: MLP

    def __init__(
        self,
        seqlen: int,
        n_heads: int,
        drop_rate: float,
        in_dim: int,
        key: PRNGKeyArray,
        strategy: Sharding,
    ):
        key1, key2, key3 = jax.random.split(key, 3)

        self.sharding = strategy(policy)

        self.seqlen = seqlen
        self.n_heads = n_heads
        self.in_dim = in_dim

        self.rope_embed = eqx.nn.RotaryPositionalEmbedding(
            embedding_size=in_dim // n_heads
        )

        self.attn_gate = eqx.nn.MultiheadAttention(
            num_heads=n_heads,
            query_size=in_dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=drop_rate,
            key=key1,
        )

        self.ln1 = eqx.nn.LayerNorm(self.in_dim)
        self.ln2 = eqx.nn.LayerNorm(self.in_dim)

        self.copy_gate = CopyGate(in_dim, drop_rate, key3, strategy)
        self.mlp = MLP(self.in_dim, self.in_dim, drop_rate, key2, strategy)

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
        mask = jnp.tril(mask)
        mask = mask * pad_mask[:, None] * pad_mask[None, :]

        return mask

    @eqx.filter_jit
    def __call__(
        self,
        inp: Float[Array, "seqlen in_dim"],
        passthrough: Float[Array, "seqlen in_dim"],
        mask: Array,
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, "seqlen in_dim"]:

        key_1, key_2, key_3 = jax.random.split(key, 3)

        inp, mask = self.sharding.shard_model_cast((inp, mask))

        scores = inp + self.attn_gate(
            query=inp,
            key_=inp,
            value=inp,
            mask=self._make_self_attention_mask(mask),
            inference=enable_dropout,
            process_heads=self.process_heads,
            key=key_1,
        )

        scores = jax.vmap(self.ln1)(scores)

        gate = self.copy_gate(scores, enable_dropout, key_2)
        ff_out = jax.nn.tanh(self.mlp(scores, enable_dropout=True, key=key_3))

        # Here we carry over the `passthrough`
        # if the gate is closed. However, in future
        # We can also carry over scores or some other repr.
        out = gate * ff_out + (1 - gate) * passthrough

        return self.sharding.shard_model_cast(out)

class AttentionBlock(eqx.Module):
    """Basic Block for LiteAttention. Uses Pre-LN from Xiong et al."""

    sharding: Sharding = eqx.field(static=True)
    seqlen: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    in_dim: int = eqx.field(static=True)

    attn_gate: eqx.nn.MultiheadAttention
    rope_embed: eqx.nn.RotaryPositionalEmbedding
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    mlp: MLP

    def __init__(
        self,
        seqlen: int,
        n_heads: int,
        drop_rate: float,
        in_dim: int,
        key: PRNGKeyArray,
        strategy: Sharding,
    ):
        key1, key2 = jax.random.split(key, 2)

        self.sharding = strategy(policy)

        self.seqlen = seqlen
        self.n_heads = n_heads
        self.in_dim = in_dim

        self.rope_embed = eqx.nn.RotaryPositionalEmbedding(
            embedding_size=in_dim // n_heads
        )

        self.attn_gate = eqx.nn.MultiheadAttention(
            num_heads=n_heads,
            query_size=in_dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=drop_rate,
            key=key1,
        )

        self.ln1 = eqx.nn.LayerNorm(self.in_dim)
        self.ln2 = eqx.nn.LayerNorm(self.in_dim)

        self.mlp = MLP(self.in_dim, self.in_dim, drop_rate, key2, strategy)

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
        mask = jnp.tril(mask)
        mask = mask * pad_mask[:, None] * pad_mask[None, :]

        return mask

    @eqx.filter_jit
    def __call__(
        self,
        inp: Float[Array, "seqlen in_dim"],
        passthrough: None, # for API compatibility
        mask: Array,
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, "seqlen in_dim"]:

        key_1, key_2 = jax.random.split(key, 2)

        inp, mask = self.sharding.shard_model_cast((inp, mask))

        x = jax.vmap(self.ln1)(inp)

        inp += self.attn_gate(
            query=x,
            key_=inp,
            value=inp,
            mask=self._make_self_attention_mask(mask),
            inference=enable_dropout,
            process_heads=self.process_heads,
            key=key_1,
        )

        x = jax.vmap(self.ln2)(inp)

        inp += self.mlp(x, enable_dropout=True, key=key_2)

        return self.sharding.shard_model_cast(inp)

class Lerp(eqx.Module):
    alpha: Array

    def __init__(self, alpha: float = 0.5):
        self.alpha = jnp.array([alpha])

    @eqx.filter_jit
    def __call__(self, x: Array, y: Array) -> Array:
        x, y = policy.cast_to_compute((x, y))

        output = self.alpha * x + (1 - self.alpha) * y

        return policy.cast_to_output(output)

class ModdedEmbedding(eqx.Module):
    '''
    Using `jnp.take` instead of naive indexing. Equinox issue #920
    '''
    sharding: Sharding = eqx.field(static=True)
    weight: Array

    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        key: PRNGKeyArray,
        strategy: Sharding
    ):
        self.sharding = strategy(policy)

        self.weight = jax.random.normal(
            key, (num_embeddings, embed_dim), dtype=jnp.float32
        ) * ((2 / (5 * embed_dim)) ** 0.5)

    def __call__(self, x: Array) -> Array:
        return jnp.take(self.weight, x, axis=0)

class UnsharedBlock(eqx.Module, Generic[L]):
    """
    Wrapper class to explicitly manage named layers that aren't
    shared between iterations.
    """

    max_iters: int
    layers: Dict[str, Tuple[L, ...]]

    def __init__(
        self,
        layers: Dict[str, Callable[[PRNGKeyArray], L]],
        max_iters: int,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, max_iters)

        # Initialize a separate instance of each 'named layer' for each iteration
        self.layers = {
            name: tuple(layer_init(keys[i]) for i in range(max_iters))
            for name, layer_init in layers.items()
        }
        self.max_iters = max_iters

    def apply_layer(self, name: str, iteration_index: int | Array, x: Array) -> Array:
        """Apply the layer for a specific iteration to input x."""
        if name not in self.layers:
            raise KeyError(f"No layer named '{name}'")

        layers = self.layers[name]

        # Use lax.switch to select the appropriate layer without indexing with tracer
        def apply_fn(i):
            def layer_apply(x):
                return layers[i](x)

            return layer_apply

        branches = tuple(apply_fn(i) for i in range(len(layers)))

        return jax.lax.switch(iteration_index, branches, x)
