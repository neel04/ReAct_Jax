from functools import partial
from typing import Optional, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .blocks import MLP, LinearProj, LiteAttention

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

    def __init__(self, seqlen: int, n_heads: int, drop_rate: float, bottleneck: int, key: PRNGKeyArray):
        key1, key2 = jax.random.split(key, 2)

        self.seqlen = seqlen
        self.n_heads = n_heads
        self.bottleneck = bottleneck

        self.attn_gate = eqx.nn.MultiheadAttention(num_heads=n_heads, query_size=bottleneck,
                                                   use_query_bias=True, use_key_bias=True,
                                                   use_value_bias=True, use_output_bias=True, 
                                                   dropout_p=drop_rate, key=key1)
        #self.attn_gate = MixerBlock(self.bottleneck, seqlen, drop_rate, key1)

        self.ln1 = eqx.nn.LayerNorm(self.bottleneck)
        self.ln2 = eqx.nn.LayerNorm(self.bottleneck)

        self.mlp = MLP(self.bottleneck, self.bottleneck, drop_rate, key2)

    def _make_self_attention_mask(
        self, pad_mask: Int[Array, " seq_len"]
    ) -> Float[Array, "num_heads seq_len seq_len"]:
        """Create self-attention mask from sequence-level mask."""
        # merge with pad_mask in the end
        mask = jnp.ones((self.seqlen, self.seqlen), dtype=jnp.bfloat16)
        mask = jnp.tril(mask)
        mask = jnp.expand_dims(mask, 0)
        return jnp.repeat(mask, self.n_heads, axis=0)
    
    def _make_mixer_mask(self, pad_mask: Array):
        # Almost same, but we triu instead of tril
        # and we don't need to merge with pad_mask
        mask = jnp.ones((self.seqlen, self.seqlen)) * pad_mask
        mask = jnp.triu(mask)
        
        return mask

    def __call__(self, x: Array, key: PRNGKeyArray, mask: Optional[Array] = None):
        # x: (seqlen, bottleneck)
        key_1, key_2 = jax.random.split(key, 2)
        
        mask = jnp.zeros_like(x) if mask is None else mask
        x = jax.vmap(self.ln1)(x)
        
        x += self.attn_gate(x, x, x,
                            mask=self._make_self_attention_mask(mask),
                            key=key_1, inference=False)
        #x += self.attn_gate(x, mask=self._make_mixer_mask(mask), key=key)
        
        x = jax.vmap(self.ln2)(x)
        x += self.mlp(x, key=key_2)

        return x

class RecurrentModule(eqx.Module):
    '''
    Bunch of AttentionBlocks
    '''
    attention_blocks: List[AttentionBlock]
    reshape_layer: eqx.Module

    def __init__(self, seqlen: int, drop_rate: float, n_heads: int,
                 num_blocks: int, bottleneck: int, key: PRNGKeyArray):  # noqa: E501
        
        keys = jax.random.split(key, num_blocks)

        self.attention_blocks = []
        
        for key in keys:
            self.attention_blocks.append(
                AttentionBlock(seqlen, n_heads, drop_rate, bottleneck * 2, key))
        
        self.reshape_layer = LinearProj(bottleneck * 2, bottleneck, key=key)

    def __call__(self, x: Float[Array, ' seqlen in_dim'], pad_mask: Array,
                 key: PRNGKeyArray) -> Float[Array, ' seqlen out_dim']:
        
        for block in self.attention_blocks:
            x = block(x, key, pad_mask)
        
        x = self.reshape_layer(x)

        return x

class output_head(eqx.Module):
    '''
    Output head for the model
    '''
    out_proj: eqx.Module

    def __init__(self, bottleneck: int, tgt_vocab_size: int, seq_len: int, key: PRNGKeyArray):
        self.out_proj = LinearProj(bottleneck, tgt_vocab_size, key=key)

    def __call__(self, x: Array) -> Array:
        x = self.out_proj(x) # (batch, seqlen, tgt_vocab_size)
        
        return x

class React(eqx.Module):
    max_iters: int = eqx.field(static=True)
    bottleneck: int = eqx.field(static=True)
    SEQLEN: int = eqx.field(static=True)

    out_head: eqx.Module
    embed_layer: eqx.nn.Embedding
    main_block: LiteAttention
    pos_enc: Array

    def __init__(self, n_heads: int, seqlen: int, max_iters: int, num_blocks: int, width: int,
                 drop_rate: float, tgt_vocab_size: int, key: PRNGKeyArray):
        
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.max_iters = max_iters
        self.bottleneck = width
        self.SEQLEN = seqlen

        src_vocab_size: int = tgt_vocab_size
        tgt_vocab_size: int = tgt_vocab_size
        drop_rate: float = drop_rate

        self.embed_layer = eqx.nn.Embedding(src_vocab_size, self.bottleneck, key=key1)

        self.pos_enc = jax.lax.stop_gradient(self.positional_encoding(self.SEQLEN, self.bottleneck))

        self.main_block = RecurrentModule(seqlen, drop_rate, n_heads, num_blocks, self.bottleneck, key=key2)

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
            return carry[0], carry[1]
        
        final_interim_thought = jax.lax.fori_loop(1, self.max_iters, main, (interim_thought, mask))  # noqa: E501
        
        return final_interim_thought

    @partial(jax.jit, static_argnames=['prev_thought', 'training'])
    def __call__(self, input: Array, iters_to_do: int, pad_mask: Array,
                 prev_thought: Optional[Array] = None, training: bool = True,
                 key: Optional[PRNGKeyArray] = None) -> Array:
        
        x = jax.vmap(self.embed_layer)(input) + self.pos_enc # (batch, seqlen, bottleneck)
        
        if isinstance(prev_thought, Array):
            interim_thought = prev_thought
        
        interim_thought, _ = self.iterate_for_steps(x, pad_mask, iters_to_do, x, key) # (batch, seqlen, bottleneck)
        
        if training:
            return self.out_head(interim_thought), interim_thought
        else:
            return self.out_head(interim_thought)