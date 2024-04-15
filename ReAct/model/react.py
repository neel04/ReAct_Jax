from typing import Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from .blocks import AttentionBlock, LinearProj, LiteAttention

class RecurrentModule(eqx.Module):
    '''
    Bunch of AttentionBlocks
    '''
    attention_blocks: PyTree[AttentionBlock]
    reshape_layer: eqx.Module

    def __init__(self,
                 seqlen: int,
                 drop_rate: float,
                 n_heads: int,
                 num_blocks: int,
                 bottleneck: int,
                 key: PRNGKeyArray):  # noqa: E501

        keys = jax.random.split(key, num_blocks)

        self.reshape_layer = LinearProj(bottleneck * 2, bottleneck, key=key)

        make_block: callable = lambda k: AttentionBlock(seqlen, n_heads, drop_rate, bottleneck, k)  # noqa: E731
        self.attention_blocks = eqx.filter(eqx.filter_vmap(make_block)(keys), eqx.is_array_like)

    def __call__(self,
                 x: Array,
                 input_arr: Array,
                 pad_mask: Array,
                 enable_dropout: bool,
                 key: PRNGKeyArray) -> Array:

        enable_dropout: bool = True
        key: PRNGKeyArray = key
        
        dynamic_part, static_part = eqx.partition(self.attention_blocks, eqx.is_array_like,
                                                  is_leaf=lambda x: isinstance(x, eqx.nn.Dropout))
        
        x = self.reshape_layer(x) # (batch, seqlen, bottleneck)
        
        def f(input_tup: Tuple[Array, int], _dynamic_bl: PyTree) -> Tuple[Tuple[Array, int], int]:
            x, idx = input_tup # i is the iteration index
            
            block = eqx.combine(_dynamic_bl, static_part) # reconstruct the block
            
            x = block(x, x, pad_mask, enable_dropout, key).astype(jnp.bfloat16)
            
            return (x, idx + 1), None

        out = eqx.internal.scan(f=f, init=(x, 0), xs=dynamic_part, kind='lax')[0][0] # throw away idx

        return out

class React(eqx.Module):
    '''
    The core ReAct model that holds utilities for performing recursive iterations
    '''
    __name__ = 'ReAct'

    max_iters: int = eqx.field(static=True)
    
    iters_weights: Array
    pos_enc: Array
    embed_layer: eqx.nn.Embedding
    main_block: LiteAttention
    post_ln: eqx.nn.LayerNorm
    out_head: eqx.Module

    def __init__(self,
                 n_heads: int,
                 seqlen: int,
                 max_iters: int,
                 num_blocks: int,
                 width: int,
                 drop_rate: float,
                 vocab_size: int,
                 key: PRNGKeyArray):

        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.max_iters = max_iters
        self.iters_weights = jnp.ones((5,), dtype=jnp.bfloat16)
        self.embed_layer = eqx.nn.Embedding(vocab_size, width, key=key1)
        self.pos_enc = jax.lax.stop_gradient(self.positional_encoding(seqlen, width))

        self.main_block = RecurrentModule(seqlen, drop_rate, n_heads, num_blocks, width, key=key2)

        self.post_ln = eqx.nn.LayerNorm(width)
        self.out_head = LinearProj(width, vocab_size, key=key4)

    def positional_encoding(self, seq_len, d_model):
        '''
        Generates the positional encoding for the input sequence
        of shape (batch_size, max_seq_len, d_model) which would be added
        to the sequence embeddings.
        '''
        position = jnp.arange(seq_len, dtype=jnp.bfloat16).reshape(-1, 1)
        div_term = jnp.exp(jnp.arange(0, d_model, 2, dtype=jnp.bfloat16) * -(jnp.log(10000.0) / d_model))
        pe = jnp.zeros((seq_len, d_model))

        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        return pe

    @eqx.filter_jit
    def iterate_for_steps(self,
                          interim_thought: Array,
                          mask: Array,
                          iters_to_do: int,
                          input_arr: Array,
                          enable_dropout: bool,
                          key: PRNGKeyArray) -> Array:

        # These are constants
        input_arr = input_arr.astype(jnp.bfloat16)
        interim_thought = interim_thought.astype(jnp.bfloat16)
        mask = mask.astype(jnp.bfloat16)

        def body_fun(thought: Array, _) -> Tuple[PyTree, Array]:
            latent = jnp.concatenate([thought, input_arr], axis=-1).astype(jnp.bfloat16)
            latent = self.main_block(latent, input_arr, mask, enable_dropout, key).astype(jnp.bfloat16)
            latent = jax.vmap(self.post_ln)(latent).astype(jnp.bfloat16)  # LN to keep scales tidy

            return latent, latent

        final_val, history = eqx.internal.scan(f=body_fun, init=interim_thought, xs=None, length=5, kind='checkpointed')
        #return jnp.einsum('i j k, i -> j k', history, self.iters_weights) # dot-product with iters_weights
        return final_val

    @eqx.filter_jit
    def __call__(self,
                 input_arr: Union[Array, Tuple[Array, Array]],
                 iters_to_do: int,
                 pad_mask: Array,
                 prev_thought: bool = False,
                 is_training: bool = True,
                 key: Optional[PRNGKeyArray] = None) -> Tuple[Array, Array]:

        if prev_thought:
            assert isinstance(input_arr, tuple), 'prev_thought is True, but input_arr is not a tuple'
            input_arr, interim_thought = input_arr
            input_arr = jax.vmap(self.embed_layer)(input_arr) + self.pos_enc # (batch, seqlen, bottleneck)
        else:
            input_arr = jax.vmap(self.embed_layer)(input_arr) + self.pos_enc # (batch, seqlen, bottleneck)
            interim_thought = input_arr.copy() # has to be a copy of the embedded + projected input array

        output = self.iterate_for_steps(interim_thought, pad_mask, iters_to_do, input_arr, is_training, key) # (batch, seqlen, bottleneck)

        return self.out_head(output), output
