import math
from typing import Any, Optional 

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from jmp import Policy

from ReAct.utils.sharding import Sharding, get_strategy

policy = Policy(compute_dtype=jnp.bfloat16, param_dtype=jnp.float32, output_dtype=jnp.bfloat16)

# ruff: noqa: F722

class NewGELU(eqx.Module):
    sharding: Sharding = eqx.field(static=True)

    def __init__(self, strategy: Sharding) -> None:
        self.sharding = strategy(policy)

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        c = math.sqrt(2.0 / math.pi)
        a = 0.044715

        x = self.sharding.shard_cast(x)

        output = 0.5 * x * (1.0 + jax.nn.tanh(c * (x + a * jnp.power(x, 3.0))))

        return self.sharding.shard_model(output)


class LinearProj(eqx.Module):
    bias: Optional[jax.Array]
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
        strategy: Sharding | Any = None
    ):
        assert (
            input_dim >= 1 or output_dim >= 1
        ), f"input_dim: {input_dim} | output_dim: {output_dim} are too small"
        wkey, bkey = jax.random.split(key, 2)

        self.sharding = strategy(policy) 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        lim = 1 / math.sqrt(input_dim)

        self.weight = self.sharding.shard_model(
            jax.random.uniform(wkey, (input_dim, output_dim), minval=-lim, maxval=lim)
            * math.sqrt(1 / (3 * input_dim))
        )

        if use_bias:
            self.bias = jax.random.uniform(bkey, (output_dim,), minval=-lim, maxval=lim)
        else:
            self.bias = jnp.zeros((output_dim,))

    def __call__(
        self,
        arr: Float[Array, "batch in_dim"],
        mask: Optional[Array] = None,
    ) -> Array:
        _mask = jnp.ones_like(self.weight) if mask is None else mask

        arr, _mask = self.sharding.shard_cast((arr, _mask))

        output = self.sharding.shard_model(
            arr @ (self.weight * _mask.astype(arr.dtype)) + self.bias
        )

        return output

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
    ):
        key1, key2 = jax.random.split(key, 2)

        self.sharding = strategy(policy)
        self.layer_1 = LinearProj(input_dim, output_dim * 4, key=key1, strategy=strategy)
        self.layer_2 = LinearProj(output_dim * 4, output_dim, key=key2, strategy=strategy)
        self.act = NewGELU(strategy)

        self.dropout = eqx.nn.Dropout(p=p)

    def __call__(self, x: Array, enable_dropout: bool, key: PRNGKeyArray):
        x = self.sharding.shard_cast(x)

        x = self.act(self.layer_1(x))

        x = self.sharding.shard_model(x)

        x = self.layer_2(x)

        output = self.dropout(x, key=key, inference=enable_dropout)

        return self.sharding.shard_model(self.act(output))


class AttentionBlock(eqx.Module):
    """Basic Block for LiteAttention"""

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

    def __call__(
        self,
        inp: Float[Array, "seqlen in_dim"],
        input_arr: Array,
        mask: Array,
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, "seqlen in_dim"]:

        key_1, key_2 = jax.random.split(key, 2)
        inp, input_arr, mask = self.sharding.shard_cast((inp, input_arr, mask))

        x = jax.vmap(self.ln1)(inp)

        inp, x = self.sharding.shard_cast((inp, x))

        inp += self.attn_gate(
            query=x,
            key_=input_arr,
            value=input_arr,
            mask=self._make_self_attention_mask(mask),
            inference=enable_dropout,
            process_heads=self.process_heads,
            key=key_1,
        )

        x = jax.vmap(self.ln2)(inp)

        inp, x = self.sharding.shard_model((inp, x))

        inp += self.mlp(x, enable_dropout=True, key=key_2)

        return self.sharding.shard_model(inp)

class Lerp(eqx.Module):
    alpha: Array

    def __init__(self, alpha: float = 0.5):
        self.alpha = jnp.array([alpha])

    def __call__(self, x: Array, y: Array) -> Array:
        x, y = policy.cast_to_compute((x, y))

        output = self.alpha * x + (1 - self.alpha) * y

        return policy.cast_to_output(output)
