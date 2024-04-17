from typing import Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from .blocks import AttentionBlock, LinearProj, LiteAttention

class RecurrentModule(eqx.Module):
    '''
    Bunch of AttentionBlocks in a pseuo-LTM fashion
    '''
    num_blocks: int = eqx.field(static=True)
    
    attention_blocks: PyTree[AttentionBlock]
    LTM_gate: AttentionBlock
    reshape_layer: LinearProj

    def __init__(self,
                 seqlen: int,
                 drop_rate: float,
                 n_heads: int,
                 num_blocks: int,
                 bottleneck: int,
                 key: PRNGKeyArray):
        
        self.num_blocks = num_blocks
        keys = jax.random.split(key, num_blocks)
        
        make_block: callable = lambda k: AttentionBlock(  # noqa: E731
            seqlen, n_heads, drop_rate, bottleneck, k
        )

        self.LTM_gate = AttentionBlock(seqlen, n_heads, drop_rate, bottleneck, key)
        self.reshape_layer = LinearProj(bottleneck * 2, bottleneck, key=key)

        self.attention_blocks = eqx.filter(eqx.filter_vmap(make_block)(keys), eqx.is_array_like)
    
    def __call__(self,
                 x: Array,
                 input_arr: Array,
                 pad_mask: Array,
                 enable_dropout: bool,
                 key: PRNGKeyArray) -> Tuple[Array, Array]:

        keys = jax.random.split(key, self.num_blocks)
        dynamic_part, static_part = eqx.partition(self.attention_blocks, eqx.is_array_like,
                                                  is_leaf=lambda x: isinstance(x, eqx.nn.Dropout))
        
        x = self.reshape_layer(x) # (seqlen, width * 2) -> (seqlen, width)
        
        def f(input_tup: Tuple[Array, int], _dynamic_bl: PyTree) -> Tuple[Tuple[Array, int], None]:
            x, idx = input_tup # i is the iteration index
            
            block = eqx.combine(_dynamic_bl, static_part) # reconstruct the block
            
            x = jax.lax.cond(idx == 0,
                             lambda: block(x, input_arr, pad_mask, enable_dropout, keys[idx]),
                             lambda: block(x, x, pad_mask, enable_dropout, keys[idx]))
            
            return (x, idx + 1), x

        out, history = eqx.internal.scan(f=f, init=(x, 0), xs=dynamic_part, kind='lax')
        
        # Handle the LTM component
        input_arr = self.LTM_gate(out[0], input_arr, pad_mask, enable_dropout, key) #TODO: Try it swapped

        return out[0], input_arr

class React(eqx.Module):
    '''
    The core ReAct model that holds utilities for performing recursive iterations
    '''
    __name__ = 'ReAct'

    max_iters: int = eqx.field(static=True)
    
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
                          input_arr: Array,
                          mask: Array,
                          iters_to_do: int,
                          enable_dropout: bool,
                          key: PRNGKeyArray) -> Array:

        # Declaring constants
        input_arr = input_arr.astype(jnp.bfloat16)
        interim_thought = interim_thought.astype(jnp.bfloat16)
        mask = mask.astype(jnp.bfloat16)

        def body_fun(carry: Tuple[Array, Array], idx: int) -> Tuple[Array, Array]:
            thought, ctx_state = carry
            
            latent = jnp.concatenate([input_arr, thought], axis=-1) # (seqlen, width * 2)
            latent, ctx_state = self.main_block(latent, ctx_state, mask, enable_dropout, key) # (seqlen, width)
            latent = jax.vmap(self.post_ln)(latent)  # Post-LN for stability 
            
            return (latent, ctx_state), ctx_state

        final_val, history = eqx.internal.scan(
            f=body_fun, init=(interim_thought, input_arr), xs=jnp.arange(5), kind="checkpointed"
        )

        return final_val[0]

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

        output = self.iterate_for_steps(interim_thought, input_arr, pad_mask, iters_to_do, is_training, key) # (batch, seqlen, bottleneck)

        return self.out_head(output), output
