from functools import partial
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float16, PRNGKeyArray

from .blocks import MLP, LinearProj, LiteAttention, NewGELU

# ruff: noqa: F722
class AttentionBlock(eqx.Module):
    """Basic Block for LiteAttention"""

    seqlen: int = eqx.field(static=True)
    activation: eqx.Module
    attn_gate: eqx.Module
    ln1: eqx.Module
    ln2: eqx.Module
    mlp: eqx.Module

    def __init__(self, seqlen: int, n_heads: int, drop_rate: float, bottleneck: int, key: PRNGKeyArray):
        key1, key2 = jax.random.split(key, 2)

        self.activation = NewGELU()
        self.seqlen = seqlen

        #self.attn_gate = LiteAttention(bottleneck, key1)
        self.attn_gate = eqx.nn.MultiheadAttention(num_heads=n_heads,
                                                   query_size=bottleneck,
                                                   use_output_bias=True, key=key1,
                                                   dropout_p=drop_rate)

        self.ln1 = eqx.nn.LayerNorm(bottleneck)
        self.ln2 = eqx.nn.LayerNorm(bottleneck)

        self.mlp = MLP(bottleneck, bottleneck, drop_rate, key2)

    def _make_self_attention_mask(self, mask_dim: int, pad_mask: Array) -> Array:
        """Create triangular self-attention causal mask. mask_dim is the head_size"""
        # 0/False -> Ignore, 1/True -> Keep
        mask = jnp.tril(jnp.ones((mask_dim, mask_dim), dtype=jnp.float32), k=0)
        mask = jnp.reshape(mask, (mask_dim, mask_dim))
        return jnp.greater(mask * pad_mask, 0.5) # Get the boolean mask

    def __call__(self, x: Array, key: PRNGKeyArray, mask: Optional[Array] = None):
        mask = jnp.zeros_like(x) if mask is None else mask
        
        x = self.ln1(x)
        #x += self.attn_gate(x)
        x += self.attn_gate(x, x, x,
                            mask=self._make_self_attention_mask(self.seqlen, mask),
                            key=key, inference=False)[0]
        x = self.ln2(x)
        x += self.mlp(x, key=key)

        return self.activation(x)  # skip connection

class RecurrentModule(eqx.Module):
    '''
    Bunch of AttentionBlocks
    '''
    gelu: eqx.Module
    reshape_layer: eqx.Module
    attention_blocks: list

    def __init__(self, seqlen: int, drop_rate: float, n_heads: int,
                 num_blocks: int, bottleneck: int, key: PRNGKeyArray):  # noqa: E501
        
        key1, key2 = jax.random.split(key)

        self.gelu = NewGELU()
        self.reshape_layer = LinearProj(bottleneck * 2, bottleneck, key=key1)

        self.attention_blocks = [
            AttentionBlock(seqlen, n_heads, drop_rate, bottleneck * 2, key2)
        ] * num_blocks

    def __call__(self, x: Float16[Array, ' seqlen in_dim'], pad_mask: Array,
                 key: PRNGKeyArray) -> Float16[Array, ' seqlen out_dim']:
        
        for block in self.attention_blocks:
            x = block(x, key, pad_mask)

        # handling recurrence
        x =  self.gelu(self.reshape_layer(x))

        return x

class output_head(eqx.Module):
    '''
    Output head for the model
    '''
    out_proj: eqx.Module
    act: eqx.Module

    def __init__(self, bottleneck: int, tgt_vocab_size: int, seq_len: int, key: PRNGKeyArray):
        # Progessively increasing the dimensionality of the output
        self.out_proj = LinearProj(bottleneck, tgt_vocab_size, key=key)
        self.act = NewGELU()

    def __call__(self, x: Array) -> Array:
        x = self.act(self.out_proj(x)) # (batch, seqlen, tgt_vocab_size)
        
        return x

class React(eqx.Module):
    max_iters: int = eqx.field(static=True)
    bottleneck: int = eqx.field(static=True)
    SEQLEN: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    input_proj: eqx.Module
    input_act: eqx.Module
    out_head: eqx.Module
    embed_layer: eqx.nn.Embedding
    main_block: LiteAttention
    id: eqx.nn.Identity
    pos_enc: jax.Array

    def __init__(self, n_heads: int, seqlen: int, max_iters: int, num_blocks: int, width: int,
                 drop_rate: float, tgt_vocab_size: int, key: PRNGKeyArray):
        
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.max_iters = max_iters
        self.bottleneck = width // 2
        self.embed_dim = self.bottleneck
        self.SEQLEN = seqlen

        src_vocab_size: int = tgt_vocab_size
        tgt_vocab_size: int = tgt_vocab_size
        drop_rate: float = drop_rate

        self.embed_layer = eqx.nn.Embedding(src_vocab_size, self.embed_dim, key=key1)
        self.input_proj = LinearProj(self.bottleneck, self.bottleneck, key=key2)
        self.input_act = NewGELU()

        self.pos_enc = jax.lax.stop_gradient(self.positional_encoding(self.SEQLEN, self.bottleneck))

        self.main_block = RecurrentModule(seqlen, drop_rate, n_heads, num_blocks, self.bottleneck, key=key2)
        self.id = eqx.nn.Identity()

        self.out_head = output_head(self.bottleneck, tgt_vocab_size, self.SEQLEN, key=key4)
    
    @partial(jax.jit, static_argnums=[1,2])
    def positional_encoding(self, seq_len, d_model):
        '''
        Generates the positional encoding for the input sequence
        of shape (batch_size, max_seq_len, d_model) which would be added
        to the sequence embeddings.
        '''
        position = jnp.arange(seq_len, dtype=jnp.float32).reshape(-1, 1)
        div_term = jnp.exp(jnp.arange(0, d_model, 2, dtype=jnp.float32) * -(jnp.log(10000.0) / d_model))
        pe = jnp.zeros((seq_len, d_model))

        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        return pe

    @partial(jax.jit, static_argnums=1)
    def iterate_for_steps(self, interim_thought: Array, mask: Array, iters_to_do: int, x: Array,
                          key: PRNGKeyArray) -> Array:
        
        def main(i: int, carry: Tuple[Array]) -> Array:
            return jax.lax.cond(i <= iters_to_do, iterate, Identity, i, carry)

        def iterate(i: int, carry: Tuple[Array]) -> Array:
            # carry[0] -> interim_thought, carry[1] -> mask
            interim_thought = jnp.concatenate([carry[0], x], 1)
            return self.main_block(interim_thought, carry[1], key), carry[1]

        def Identity(i: int, carry: Array) -> Array:
            return self.id(carry[0]), self.id(carry[1])

        final_interim_thought = jax.lax.fori_loop(1, self.max_iters, main, (interim_thought, mask))  # noqa: E501
        return final_interim_thought

    @partial(jax.jit, static_argnames=['prev_thought', 'training'])
    def __call__(self, input: Array, iters_to_do: int, pad_mask: Array,
                 prev_thought: Optional[Array] = None, training: bool = True,
                 key: Optional[PRNGKeyArray] = None) -> Array:
        
        x = jax.vmap(self.embed_layer)(input) + self.pos_enc # (batch, seqlen, embed_dim
        interim_thought = self.input_act(self.input_proj(x)) # (batch, seqlen, bottleneck)

        if isinstance(prev_thought, Array):
            interim_thought = prev_thought
        
        interim_thought, _ = self.iterate_for_steps(interim_thought, pad_mask, iters_to_do, x, key) # (batch, seqlen, bottleneck)
        
        if training:
            return self.out_head(interim_thought), interim_thought
        else:
            return self.out_head(interim_thought)