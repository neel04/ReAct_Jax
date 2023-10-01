import jax
import equinox as eqx
import jax.numpy as jnp
import math

from jaxtyping import Array, Float16, PRNGKeyArray
from typing import Optional

# ruff: noqa: F722

class NewGELU(eqx.Module):
	def __call__(self, x: jax.Array) -> jax.Array:
		c: float = math.sqrt(2.0 / math.pi)
		a: float = 0.044715
		return 0.5 * x * (1.0 + jax.nn.tanh(c * (x + a * jnp.power(x, 3.0))))

class MLP(eqx.Module):
    '''A simple MLP - w/ Dropout'''
    layers: eqx.nn.Sequential
    dropout: eqx.nn.Dropout

    def __init__(self, input_dim: int, output_dim: int, p: float, key: PRNGKeyArray):
        key1, key2 = jax.random.split(key, 2)

        self.layers = [
            LinearProj(input_dim, output_dim, key=key1),
            eqx.nn.Lambda(NewGELU()),
            LinearProj(output_dim, output_dim, key=key2),
            ]

        self.dropout = eqx.nn.Dropout(p=p)

<<<<<<< HEAD
    def __call__(self, x: Array, key: PRNGKeyArray):
        for layer in self.layers:
            x = layer(x)
        
        return self.dropout(x, key=key, inference=False)
=======
    def __call__(self, x: Array, key: Optional[PRNGKeyArray] = None):
        for layer in self.layers:
            x = layer(x)
        
        if key is None:
            return self.dropout(x)
        else:
            return self.dropout(x, key=key, inference=False)
>>>>>>> main

class LinearProj(eqx.Module):

	bias: Optional[jax.Array]
	weight: jax.Array

	input_dim: int = eqx.field(static=True)
	output_dim: int = eqx.field(static=True)
	use_bias: bool = eqx.field(static=True)

	def __init__(self, input_dim, output_dim, key: PRNGKeyArray, use_bias=True):
		assert input_dim >= 1 or output_dim >= 1, f'input_dim: {input_dim} | output_dim: {output_dim} are too small'
		wkey, bkey = jax.random.split(key, 2)

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.use_bias = use_bias

		lim = 1 / math.sqrt(input_dim)
		self.weight = jax.random.uniform(bkey, (input_dim, output_dim), minval=-lim, maxval=lim)

		if use_bias:
			self.bias = jax.random.uniform(wkey, (output_dim,), minval=-lim, maxval=lim)
		else:
			self.bias = jnp.zeros((output_dim,))

	@jax.jit
	def __call__(self, input: Float16[Array, 'batch in_dim']):
		return input @ self.weight + self.bias

class LiteAttention(eqx.Module):
	input_dim: int = eqx.field(static=True)
	weight: eqx.Module

	def __init__(self, input_dim: int, key: PRNGKeyArray):
		self.input_dim = input_dim
		self.weight = LinearProj(input_dim, input_dim, use_bias=False, key=key)

	@jax.jit
	def __call__(self, x: Float16[Array, 'batch in_dim']):
		attn_weights = jax.nn.softmax(self.weight(x), axis=1) # type: ignore
		return x * attn_weights

if __name__ == '__main__':
	key = jax.random.PRNGKey(0)
	LA = LiteAttention(256, key)
	test = jax.random.normal(key, (128, 256))
	print(LA(test).shape)