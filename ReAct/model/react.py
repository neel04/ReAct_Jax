from typing import List, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from .blocks import AttentionBlock, LinearProj, LiteAttention

class RecurrentModule(eqx.Module):
    '''
    Bunch of AttentionBlocks
    '''
    attention_blocks: List[AttentionBlock]
    reshape_layer: eqx.Module

    def __init__(self,
                 seqlen: int,
                 drop_rate: float,
                 n_heads: int,
                 num_blocks: int, 
                 bottleneck: int, 
                 key: PRNGKeyArray):  # noqa: E501
        
        keys = jax.random.split(key, num_blocks)
        
        self.attention_blocks = []
        self.reshape_layer = LinearProj(bottleneck * 2, bottleneck, key=key)
        
        for key in keys:
            self.attention_blocks.append(
                AttentionBlock(seqlen, n_heads, drop_rate, bottleneck, key))
        
    def __call__(self, x: Array,
                 input_arr: Array,
                 pad_mask: Array,
                 enable_dropout: bool,
                 key: PRNGKeyArray) -> Array:
        
        x = self.reshape_layer(x) # (batch, seqlen, bottleneck)
        
        for idx, block in enumerate(self.attention_blocks):
            if idx == 0:
                # cross attention with input_arr
                x = block(x, input_arr, pad_mask, enable_dropout, key).astype(jnp.bfloat16)
            else:
                # self attention with input_arr
                x = block(x, x, pad_mask, enable_dropout, key).astype(jnp.bfloat16)
        
        return x

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
        self.iters_weights = jax.random.normal(key, (5,))
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
        
        def body_fun(carry: Array, _) -> Tuple[PyTree, Array]:
            thought, mask = carry
            
            latent = jnp.concatenate([thought, input_arr], axis=-1).astype(jnp.bfloat16)
            latent = self.main_block(latent, input_arr, mask, enable_dropout, key).astype(jnp.bfloat16)
            latent = jax.vmap(self.post_ln)(latent).astype(jnp.bfloat16)  # LN to keep scales tidy

            return (latent, mask), latent

        final_val, _ = eqx.internal.scan(f=body_fun, init=(interim_thought, mask), xs=None, length=5, kind='checkpointed')
        
        #return jnp.dot(history, self.iters_weights)
        return final_val[0]

    @eqx.filter_jit
    def __call__(self,
                 input_arr: Union[Array, Tuple[Array]],
                 iters_to_do: int,
                 pad_mask: Array,
                 prev_thought: bool = False,
                 is_training: bool = True,
                 key: Optional[PRNGKeyArray] = None) -> Tuple[Array]:
        
        if prev_thought:
            assert isinstance(input_arr, tuple), 'prev_thought is True, but input_arr is not a tuple'
            input_arr, interim_thought = input_arr
            input_arr = jax.vmap(self.embed_layer)(input_arr) + self.pos_enc # (batch, seqlen, bottleneck)
        else:
            input_arr = jax.vmap(self.embed_layer)(input_arr) + self.pos_enc # (batch, seqlen, bottleneck)
            interim_thought = input_arr.copy() # has to be a copy of the embedded + projected input array
        
        output = self.iterate_for_steps(interim_thought, pad_mask, iters_to_do, input_arr, is_training, key) # (batch, seqlen, bottleneck)
        
        return self.out_head(output), output