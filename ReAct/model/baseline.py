from typing import Callable, List

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree
from jmp import Policy

from .blocks import AttentionBlock, LinearProj

policy = Policy(compute_dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, output_dtype=jnp.bfloat16)

class main_block(eqx.Module):
    '''
    Main block of the GPT model where you just compute
    all the attention blocks sequentially
    '''

    attention_blocks: List[AttentionBlock]
    
    def __init__(self,
                 seqlen: int,
                 bottleneck: int,
                 n_heads: int,
                 drop_rate: float,
                 num_blocks: int,
                 key: PRNGKeyArray):
        
        keys = jax.random.split(key, num_blocks)
        make_block: Callable = lambda k: AttentionBlock(seqlen, n_heads, drop_rate, bottleneck, k)  # noqa: E731
        
        # weights have dim: (num_blocks, *)
        self.attention_blocks = eqx.filter(eqx.filter_vmap(make_block)(keys), eqx.is_array_like)
    
    def __call__(self,
                 input_arr: Array,
                 pad_mask: Array,
                 enable_dropout: bool,
                 key: PRNGKeyArray) -> Array:
        
        enable_dropout: bool = True
        key: PRNGKeyArray = key
        
        # static_part are activations etc.
        # dynamic_part are the parameters
        dynamic_part, static_part = eqx.partition(self.attention_blocks, eqx.is_array_like,
                                                  is_leaf=lambda x: isinstance(x, eqx.nn.Dropout))
        
        def f(input_arr: Array, _dynamic_bl: PyTree):
            block = eqx.combine(_dynamic_bl, static_part)
            output = block(input_arr, input_arr, pad_mask, enable_dropout, key)

            return policy.cast_to_output(output), None

        out, _ = jax.lax.scan(f=f, init=input_arr, xs=dynamic_part, unroll=2)
        
        return out
        
class GPT(eqx.Module):
    '''
    Vanilla Transformer model
    '''
    __name__ = 'GPT'
    
    embed_layer: eqx.Module
    pos_enc: Array
    main_block: eqx.Module
    out_head: eqx.Module
    
    def __init__(self,
                 n_heads: int,
                 seqlen: int,
                 num_blocks: int,
                 width: int,
                 drop_rate: float,
                 vocab_size: int,
                 key: PRNGKeyArray):
        
        keys = jax.random.split(key, 3)
        
        self.embed_layer = eqx.nn.Embedding(vocab_size, width, key=keys[0])
        self.pos_enc = jax.lax.stop_gradient(self.positional_encoding(seqlen, width))

        self.main_block = main_block(seqlen, width, n_heads, drop_rate, num_blocks, key=keys[1])
        self.out_head = LinearProj(width, vocab_size, key=keys[2])
    
    def positional_encoding(self, seq_len: int, d_model: int):
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
    
    @eqx.filter_jit
    def __call__(self,
                 input_arr: Array,
                 pad_mask: Array,
                 enable_dropout: bool,
                 key: PRNGKeyArray) -> Array:
        
        input_arr = jax.vmap(self.embed_layer)(input_arr) + self.pos_enc

        input_arr, pad_mask = policy.cast_to_compute((input_arr, pad_mask))
        output = self.main_block(input_arr, pad_mask, enable_dropout, key)
        output = self.main_block(output, pad_mask, enable_dropout, key)
        
        return self.out_head(output)

if __name__ == '__main__':
    # Testing the model
    bsz: int = 128
    key = jax.random.PRNGKey(69)
    
    input_arr = jnp.ones((bsz, 512), dtype=jnp.int32) # (batch_size, seq_len)
    pad_mask = jnp.ones((bsz, 512), dtype=jnp.int32)
    
    mygpt = GPT(n_heads=4,
                seqlen=512,
                num_blocks=4,
                width=64,
                drop_rate=0.1,
                vocab_size=50257,
                key=key)
    
    output = jax.vmap(mygpt, (0, 0, None))(input_arr, pad_mask, key)
    print(f'Output shape: {output.shape}')