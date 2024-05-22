from functools import partial
from typing import Optional, Tuple, Union, List

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree, Int
from jmp import Policy

from .blocks import MLP, AttentionBlock, LinearProj, LiteAttention, NewGELU

policy = Policy(compute_dtype=jnp.bfloat16, param_dtype=jnp.float32, output_dtype=jnp.bfloat16)

# ruff: noqa: E402, E731

class RecurrentModule(eqx.Module):
    '''
    Bunch of Attentionlayers in a pseuo-LSTM fashion
    '''
    num_layers: int = eqx.field(static=True)

    attention_layers: Array
    reshape_layers: List
    initial_layer: eqx.Module
    hist_gate: eqx.Module
    act: eqx.Module
    ctx_gate: eqx.Module

    def __init__(self,
                 seqlen: int,
                 drop_rate: float,
                 n_heads: int,
                 num_layers: int,
                 bottleneck: int,
                 key: PRNGKeyArray):
        
        keys = jax.random.split(key, num_layers)
        in_dims, out_dims = self.generate_dims(bottleneck, num_layers)

        self.act = NewGELU()
        self.num_layers = num_layers

        self.reshape_layers = eqx.nn.Sequential([
            LinearProj(in_dim, out_dim, key=key)
            for in_dim, out_dim in zip(out_dims, in_dims[::-1])
        ])

        self.initial_layer = MLP(bottleneck * 2, bottleneck, p=0.0, key=keys[0])
        self.hist_gate = LinearProj(bottleneck * 2, bottleneck, key=keys[1])
        self.ctx_gate = MLP(bottleneck, bottleneck, p=0.0, key=keys[2])

        make_layer: callable = partial(self.make_layer, seqlen, n_heads, drop_rate)

        self.attention_layers = [
            make_layer(bottleneck, key)
            for bottleneck, key in zip(out_dims, keys)
        ]

    @staticmethod
    def make_layer(seqlen: int, n_heads: int, drop_rate: float, bottleneck: int, key: PRNGKeyArray) -> AttentionBlock:
        return AttentionBlock(seqlen, n_heads, drop_rate, bottleneck, key=key)

    @staticmethod
    def generate_dims(bottleneck: int, num_layers: int) -> List[int]:
        out_dims = [bottleneck // (2**i) for i in range(num_layers // 2)]
        out_dims += out_dims[::-1] if num_layers % 2 == 0 else out_dims[:-1][::-1]
        in_dims = [bottleneck] + out_dims[:-1]

        return in_dims, out_dims
    
    @staticmethod
    def unroll_scan(f: callable, init: Tuple, xs):
        carry = init
        ys = []
        
        for x in xs:
            carry, y = f(carry, x)
            ys.append(y)

        # Pad the output tensors to the same rank
        max_rank = max([y.shape[-1] for y in ys])
        ys = [
            jnp.pad(y, ((0, 0),) * (y.ndim - 1) + ((0, max_rank - y.shape[-1]),))
            for y in ys
        ]

        return carry, jnp.stack(ys)
    
    def __call__(self,
                 x: Array,
                 ctx_state: Array,
                 pad_mask: Array,
                 enable_dropout: bool,
                 key: PRNGKeyArray) -> Tuple[Array, Array]:

        keys = jax.random.split(key, self.num_layers)
        
        x = self.initial_layer(x, enable_dropout, key)
        
        def f(input_tup: Tuple[Array, int], layer: callable) -> Tuple[Tuple[Array, int], Array]:
            x, idx = input_tup
            
            if idx == 0:
                x = layer(x, ctx_state, pad_mask, enable_dropout, keys[idx])
            else:
                x = layer(x, x, pad_mask, enable_dropout, keys[idx])
            
            # Up/Down-sampling
            x = self.act(self.reshape_layers[idx](x))
            
            return (x, idx + 1), x

        out, history = self.unroll_scan(f=f, init=(x, 0), xs=self.attention_layers)

        hist_lerp = self.act(self.hist_gate(jnp.concat([history.mean(0), ctx_state], axis=-1)))
        ctx_state = self.ctx_gate(hist_lerp, enable_dropout=enable_dropout, key=key)

        return out[0], ctx_state

class React(eqx.Module):
    '''
    The core ReAct model that holds utilities for performing recursive iterations
    '''
    __name__ = 'ReAct'

    max_iters: int = eqx.field(static=True)
    
    alpha: float
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
        self.alpha = jnp.array([0.5])

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
    def iterate_for_steps(self,
                          interim_thought: Array,
                          input_arr: Array,
                          mask: Array,
                          iters_to_do: int,
                          enable_dropout: bool,
                          key: PRNGKeyArray) -> Array:

        @eqx.filter_jit
        def body_fun(carry: Tuple[Array, Array], idx: int) -> Tuple[Tuple, Array]:
            thought, ctx_state = carry
            
            latent = jnp.concatenate([input_arr, thought], axis=-1) # (seqlen, width * 2)
            latent, ctx_state = self.main_block(latent, ctx_state, mask, enable_dropout, key) # (seqlen, width)
            latent = jax.vmap(self.post_ln)(latent)  # Post-LN for stability 

            latent = policy.cast_to_output(latent) # mixed precision
            
            return (latent, ctx_state), latent

        final_val, history = eqx.internal.scan(
            f=body_fun, init=(interim_thought, input_arr), xs=jnp.arange(iters_to_do), kind='checkpointed'
        )

        return self.alpha * final_val[0] + (1 - self.alpha) * history.mean(0)

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

        input_arr, interim_thought = policy.cast_to_compute((input_arr, interim_thought))

        output = self.iterate_for_steps(interim_thought, input_arr, pad_mask, iters_to_do, is_training, key) # (batch, seqlen, bottleneck)

        return self.out_head(output), output
