from functools import partial
from typing import Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree
from jmp import Policy

from .blocks import MLP, LinearProj, LiteAttention, AttentionBlock, lerp

policy = Policy(compute_dtype=jnp.bfloat16, param_dtype=jnp.float32, output_dtype=jnp.bfloat16)

# ruff: noqa: E402, E731

class RecurrentModule(eqx.Module):
    '''
    Bunch of Attentionlayers in a pseuo-LSTM fashion
    '''
    num_layers: int = eqx.field(static=True)

    attention_layers: Array
    reshape_layer: eqx.Module
    history_weight: Array
    ctx_gate: eqx.Module

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

        self.num_layers = num_layers

        self.reshape_layer = LinearProj(bottleneck * 2, bottleneck, key=keys[0])
        self.ctx_gate = MLP(bottleneck, bottleneck, p=drop_rate, key=keys[1])
        self.history_weight = jnp.ones(num_layers, dtype=jnp.bfloat16)

        make_layer: callable = lambda k: self.make_layer(
            seqlen, n_heads, drop_rate, bottleneck, k
        )

        self.attention_layers = eqx.filter(eqx.filter_vmap(make_layer)(keys), eqx.is_array_like)

    @staticmethod
    def make_layer(seqlen: int, n_heads: int, drop_rate: float, bottleneck: int, key: PRNGKeyArray) -> AttentionBlock:
        return AttentionBlock(seqlen, n_heads, drop_rate, bottleneck, key)

    def __call__(
        self,
        x: Array,
        ctx_state: Array,
        pad_mask: Array,
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Tuple[Array, Array]:

        keys = jax.random.split(key, self.num_layers)
        dynamic_part, static_part = eqx.partition(self.attention_layers, eqx.is_array_like,
                                                  is_leaf=lambda x: isinstance(x, eqx.nn.Dropout))
        
        x = policy.cast_to_compute(self.reshape_layer(x)) # (seqlen, bottleneck)
        
        def f(input_tup: Tuple[Array, int], _dynamic_bl: PyTree) -> Tuple[Tuple[Array, int], Array]:
            x, idx = input_tup
            layer = eqx.combine(_dynamic_bl, static_part) # reconstruct the layer

            x = jax.lax.cond(idx == 0,
                             lambda : layer(x, ctx_state, pad_mask, enable_dropout, keys[idx]),
                             lambda : layer(x, x, pad_mask, enable_dropout, keys[idx]))
            
            ctx = self.ctx_gate(x, enable_dropout=enable_dropout, key=keys[idx])
            
            return (x, idx + 1), ctx

        out, ctx_history = jax.lax.scan(f=f, init=(x, 0), xs=dynamic_part, unroll=True)
        ctx_state += jnp.einsum('i j k, i -> j k', ctx_history, self.history_weight)

        return out[0], ctx_state

class React(eqx.Module):
    '''
    The core ReAct model that holds utilities for performing recursive iterations
    '''
    __name__ = 'ReAct'

    max_iters: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    
    pos_enc: Array
    embed_layer: eqx.nn.Embedding
    iteration_index_pe: eqx.nn.Embedding
    main_block: LiteAttention
    post_ln: eqx.nn.LayerNorm
    out_head: eqx.Module
    ctx_mixing: eqx.Module

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
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.max_iters = max_iters
        self.width = width

        self.embed_layer = eqx.nn.Embedding(vocab_size, width, key=key1)
        self.iteration_index_pe = eqx.nn.Embedding(max_iters, width, key=key2)
        self.pos_enc = jax.lax.stop_gradient(self.positional_encoding(seqlen, width))

        self.main_block = RecurrentModule(seqlen, drop_rate, n_heads, num_blocks, width, key=key3)
        self.ctx_mixing = MLP(width * 2, width, p=drop_rate, key=key4)

        self.post_ln = eqx.nn.LayerNorm(width)
        self.out_head = LinearProj(width, vocab_size, key=key4)

    def positional_encoding(self, seq_len, d_model):
        '''
        Generates the positional encoding for the input sequence
        of shape (batch_size, max_seq_len, d_model) which would be added
        to the sequence embeddings.
        '''
        position = jnp.arange(seq_len).reshape(-1, 1)
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
        pe = jnp.zeros((seq_len, d_model))

        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        return pe

    @partial(jax.jit, static_argnums=(4, 5, 6))
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

        @eqx.filter_jit
        def body_fun(carry: Tuple[Array, Array], idx: int) -> Tuple[Tuple, Array]:
            thought, ctx_state = carry

            ctx_state = lerp()(ctx_state, self.iteration_index_pe(idx))

            latent = jnp.concatenate([input_arr, thought], axis=-1)  # (seqlen, width * 2)
            out_latent, ctx_state = self.main_block(latent, ctx_state, mask, enable_dropout, key)  # (seqlen, width)
            latent = jax.vmap(self.post_ln)(out_latent)  # Post-LN for stability

            latent = policy.cast_to_output(latent) # mixed precision
            
            return (latent, ctx_state), latent

        final_val, history = eqx.internal.scan(
            f=body_fun, init=(interim_thought, input_arr), xs=jnp.arange(iters_to_do), kind='checkpointed'
        )

        output = jnp.concatenate([final_val[0], final_val[1]], axis=-1)  # (seqlen, width * 2)

        return self.ctx_mixing(output, enable_dropout, key)

    @eqx.filter_jit
    def __call__(
        self,
        input_arr: Union[Array, Tuple[Array, Array]],
        iters_to_do: int,
        pad_mask: Array,
        prev_thought: bool = False,
        is_training: bool = True,
        key: Optional[PRNGKeyArray] = None,
    ) -> Tuple[Array, Array]:

        if prev_thought:
            assert isinstance(input_arr, tuple), 'prev_thought is True, but input_arr is not a tuple'
            input_arr, interim_thought = input_arr
            input_arr = jax.vmap(self.embed_layer)(input_arr) + self.pos_enc # (batch, seqlen, bottleneck)
        else:
            input_arr = jax.vmap(self.embed_layer)(input_arr) + self.pos_enc # (batch, seqlen, bottleneck)
            interim_thought = input_arr.copy() # has to be a copy of the embedded + projected input array

        input_arr, interim_thought = policy.cast_to_compute((input_arr, interim_thought))

        output = self.iterate_for_steps(interim_thought, input_arr, pad_mask, iters_to_do, is_training, key) # (batch, seqlen, bottleneck)

        return self.out_head(output), output
