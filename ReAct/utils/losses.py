# Copyright 2023 The T5X Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loss functions."""

import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

VOCAB_SIZE = 50304

@jax.custom_vjp
def cross_entropy_with_logits(
    logits: Array, targets: Array, z_loss: float = 1e-4
) -> Tuple[Array, Array]:
    """Computes cross entropy loss with stable custom gradient.

    Computes a stabilized-gradient version of:
      -jnp.sum(targets * nn.log_softmax(logits), axis=-1)

    If z_loss > 0, then an auxiliary loss equal to z_loss*log(z)^2
    will be added to the cross entropy loss (z = softmax normalization constant).
    The two uses of z_loss are:
    1. To keep the logits from drifting too far from zero, which can cause
       unacceptable roundoff errors in bfloat16.
    2. To encourage the logits to be normalized log-probabilities.

    Args:
      logits: [batch, length, num_classes] float array.
      targets: categorical one-hot targets [batch, length, num_classes] float
        array.
      z_loss: coefficient for auxilliary z-loss loss term.

    Returns:
      tuple with the total loss and the z_loss, both
      float arrays with shape [batch, length].
    """
    logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    log_softmax = logits - logits_sum
    loss = -jnp.sum(targets * log_softmax, axis=-1)
    # Add auxilliary z-loss term.
    log_z = jnp.squeeze(logits_sum, axis=-1)
    total_z_loss = z_loss * jax.lax.square(log_z)
    loss += total_z_loss
    return loss, total_z_loss


def _cross_entropy_with_logits_fwd(
    logits: Array, targets: Array, z_loss: float = 0.0
) -> Tuple[
    Tuple[Array, Array],
    Tuple[Array, ...],
]:
    """Forward-mode of `cross_entropy_with_logits`."""
    max_logit = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_logit
    exp_shifted = jnp.exp(shifted)
    sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
    log_softmax = shifted - jnp.log(sum_exp)
    loss = -jnp.sum(targets * log_softmax, axis=-1)
    # Add auxilliary z-loss term.
    log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
    total_z_loss = z_loss * jax.lax.square(log_z)
    loss += total_z_loss
    return (loss, total_z_loss), (
        logits,
        targets,
        z_loss,
        exp_shifted,
        sum_exp,  # pytype: disable=bad-return-type  # jax-ndarray
        log_softmax,
        log_z,
    )


def _cross_entropy_with_logits_bwd(
    res: Tuple[Array, ...],
    g: Tuple[Array, Array],
) -> Tuple[Array, Array, Array]:
    """Backward-mode of `cross_entropy_with_logits`."""
    g = g[0]  # Ignore z_loss component as that is only used for logging.
    logits, targets, z_loss, exp_shifted, sum_exp, log_softmax, log_z = res
    # z-loss term adds the (2 * z_loss * log_z) factor.
    deriv = (
        jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp - targets
    )
    g_logits = jnp.expand_dims(g, axis=-1) * deriv
    g_targets = -jnp.expand_dims(g, axis=-1) * log_softmax
    return (
        jnp.asarray(g_logits, logits.dtype),
        jnp.asarray(g_targets, targets.dtype),
        jnp.array(0.0),
    )  # sets z-loss coeff gradient to 0


cross_entropy_with_logits.defvjp(
    _cross_entropy_with_logits_fwd, _cross_entropy_with_logits_bwd
)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def cross_entropy_with_lm_head_blockwise(
    embeddings: Array,
    lm_head_weight: Array,
    lm_head_bias: Array,
    labels: Array,
    z_loss: float = 1e-4,
    block_size: int = 4096,
    dtype: Optional[jnp.dtype] = jnp.float32,
) -> Tuple[Array, Array]:
    """Blockwise CE that consumes embeddings and LM head weights."""

    loss, z_loss_mean, _ = _cross_entropy_with_lm_head_blockwise_impl(
        embeddings,
        lm_head_weight,
        lm_head_bias,
        labels,
        z_loss=z_loss,
        block_size=block_size,
        dtype=dtype,
    )
    return loss, z_loss_mean


def _cross_entropy_with_lm_head_blockwise_impl(
    embeddings: Array,  # per-token hidden states
    lm_head_weight: Array,
    lm_head_bias: Array,
    labels: Array,
    z_loss: float,
    block_size: int,
    dtype: Optional[jnp.dtype],
) -> Tuple[Array, Array, Array]:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    block_size = min(block_size, VOCAB_SIZE)

    num_blocks = VOCAB_SIZE // block_size  # tile over vocab size
    remainder = VOCAB_SIZE - num_blocks * block_size

    compute_dtype = (
        dtype
        if dtype is not None
        else jnp.result_type(embeddings, lm_head_weight, lm_head_bias)
    )

    target_logit = jnp.zeros(labels.shape, dtype=compute_dtype)
    logsumexp = jnp.full(labels.shape, -jnp.inf, dtype=compute_dtype)
    max_logit = jnp.full(labels.shape, -jnp.inf, dtype=compute_dtype)

    def process_block(block_idx: int, acc: Tuple[Array, ...], current_block_size: int):
        target_logit_prev, logsumexp_prev, max_prev = acc

        start = block_idx * block_size

        w_b = jax.lax.dynamic_slice(
            lm_head_weight, (0, start), (lm_head_weight.shape[0], current_block_size)
        )

        b_b = jax.lax.dynamic_slice(lm_head_bias, (start,), (current_block_size,))

        logits_b = jnp.einsum("bsh,hv->bsv", embeddings, w_b) + b_b

        if logits_b.dtype != compute_dtype:
            logits_b = logits_b.astype(compute_dtype)

        block_max = jnp.max(logits_b, axis=-1)
        max_new = jnp.maximum(max_prev, block_max)

        # max-shift trick for stability
        # Stable streaming logsumexp across blocks using the new global max.
        logsumexp_new = max_new + jnp.log(
            jnp.exp(logsumexp_prev - max_new)
            + jnp.sum(jnp.exp(logits_b - max_new[..., None]), axis=-1)
        )

        labels_in_block = labels - start
        in_block = (labels_in_block >= 0) & (labels_in_block < current_block_size)
        safe_labels = jnp.where(in_block, labels_in_block, 0)

        gathered = jnp.take_along_axis(
            logits_b, safe_labels[..., None], axis=-1
        ).squeeze(-1)

        # Add only the target logit from the block that contains the label.
        target_logit_new = target_logit_prev + jnp.where(
            in_block, gathered, jnp.zeros_like(gathered)
        )

        return target_logit_new, logsumexp_new, max_new

    def body(block_idx: int, acc):
        return process_block(block_idx, acc, current_block_size=block_size)

    target_logit, logsumexp, max_logit = jax.lax.fori_loop(
        0, num_blocks, body, (target_logit, logsumexp, max_logit)
    )

    if remainder:
        target_logit, logsumexp, _ = process_block(
            num_blocks, (target_logit, logsumexp, max_logit), remainder
        )

    loss_sum = jnp.sum(logsumexp - target_logit)
    z_loss_sum = jnp.sum(z_loss * jax.lax.square(logsumexp))
    loss_sum = loss_sum + z_loss_sum

    denom = jnp.array(embeddings.shape[0] * embeddings.shape[1], dtype=compute_dtype)

    return loss_sum / denom, z_loss_sum / denom, logsumexp


def _cross_entropy_with_lm_head_blockwise_fwd(
    embeddings: Array,
    lm_head_weight: Array,
    lm_head_bias: Array,
    labels: Array,
    z_loss: float,
    block_size: int,
    dtype: Optional[jnp.dtype],
) -> Tuple[
    Tuple[Array, Array],
    Tuple[Array, Array, Array, Array, Array],
]:
    loss, z_loss_mean, logsumexp = _cross_entropy_with_lm_head_blockwise_impl(
        embeddings,
        lm_head_weight,
        lm_head_bias,
        labels,
        z_loss=z_loss,
        block_size=block_size,
        dtype=dtype,
    )
    return (loss, z_loss_mean), (
        embeddings,
        lm_head_weight,
        lm_head_bias,
        labels,
        logsumexp,
    )


def _cross_entropy_with_lm_head_blockwise_bwd(
    z_loss: float,
    block_size: int,
    dtype: Optional[jnp.dtype],
    res: Tuple[Array, Array, Array, Array, Array],
    g: Tuple[Array, Array],
) -> Tuple[Array, Array, Array, None]:
    g_loss, _ = g
    embeddings, lm_head_weight, lm_head_bias, labels, logsumexp = res

    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    block_size = min(block_size, VOCAB_SIZE)

    num_blocks = VOCAB_SIZE // block_size
    remainder = VOCAB_SIZE - num_blocks * block_size

    compute_dtype = (
        dtype
        if dtype is not None
        else jnp.result_type(embeddings, lm_head_weight, lm_head_bias)
    )

    if g_loss is None:
        g_scale = jnp.array(0.0, dtype=compute_dtype)
    else:
        g_scale = jnp.asarray(g_loss, dtype=compute_dtype)

    denom = jnp.array(embeddings.shape[0] * embeddings.shape[1], dtype=compute_dtype)
    g_scale = g_scale / denom

    grad_embeddings = jnp.zeros_like(embeddings, dtype=compute_dtype)
    grad_weight = jnp.zeros_like(lm_head_weight, dtype=compute_dtype)
    grad_bias = jnp.zeros_like(lm_head_bias, dtype=compute_dtype)

    # d/dlogsumexp of (logsumexp + z_loss * logsumexp^2) term.
    scale = g_scale * (1 + 2 * z_loss * logsumexp)

    def process_block(block_idx: int, acc, current_block_size: int):
        grad_embeddings_prev, grad_weight_prev, grad_bias_prev = acc

        start = block_idx * block_size
        w_b = jax.lax.dynamic_slice(
            lm_head_weight, (0, start), (lm_head_weight.shape[0], current_block_size)
        )
        b_b = jax.lax.dynamic_slice(lm_head_bias, (start,), (current_block_size,))

        logits_b = jnp.einsum("bsh,hv->bsv", embeddings, w_b) + b_b

        if logits_b.dtype != compute_dtype:
            logits_b = logits_b.astype(compute_dtype)

        # Softmax probs for this vocab block using full-vocab logsumexp.
        p_b = jnp.exp(logits_b - logsumexp[..., None])

        labels_in_block = labels - start
        in_block = (labels_in_block >= 0) & (labels_in_block < current_block_size)
        safe_labels = jnp.where(in_block, labels_in_block, 0)
        target_b = (
            jax.nn.one_hot(safe_labels, current_block_size, dtype=compute_dtype)
            * in_block[..., None]
        )

        # dL/dlogits for this block: scaled probs minus one-hot target.
        d_loss = scale[..., None] * p_b - g_scale * target_b

        # Accumulate grads for embeddings and head parameters from this block.
        grad_embeddings_b = jnp.einsum(
            "bsv,hv->bsh", d_loss, w_b
        )  # broadcasted (d_loss @ w_b.T)
        grad_weight_b = jnp.einsum(
            "bsh,bsv->hv", embeddings, d_loss
        )  # broadcasted (embeddings.T @ d_loss)
        grad_bias_b = jnp.sum(d_loss, axis=(0, 1))

        grad_embeddings_new = grad_embeddings_prev + grad_embeddings_b
        grad_weight_new = jax.lax.dynamic_update_slice(
            grad_weight_prev, grad_weight_b, (0, start)
        )
        grad_bias_new = jax.lax.dynamic_update_slice(
            grad_bias_prev, grad_bias_b, (start,)
        )

        return grad_embeddings_new, grad_weight_new, grad_bias_new

    def body(block_idx: int, acc):
        return process_block(block_idx, acc, current_block_size=block_size)

    grad_embeddings, grad_weight, grad_bias = jax.lax.fori_loop(
        0, num_blocks, body, (grad_embeddings, grad_weight, grad_bias)
    )

    if remainder:
        grad_embeddings, grad_weight, grad_bias = process_block(
            num_blocks, (grad_embeddings, grad_weight, grad_bias), remainder
        )

    return (
        jnp.asarray(grad_embeddings, embeddings.dtype),
        jnp.asarray(grad_weight, lm_head_weight.dtype),
        jnp.asarray(grad_bias, lm_head_bias.dtype),
        None,
    )


cross_entropy_with_lm_head_blockwise.defvjp(
    _cross_entropy_with_lm_head_blockwise_fwd,
    _cross_entropy_with_lm_head_blockwise_bwd,
)
