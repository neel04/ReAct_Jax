from typing import Callable, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import alias, base, combine, numerics, transform, utils


# Helper function to contain the original 2D orthogonalization logic
def _orthogonalize_2d_logic(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-8,
) -> jax.Array:
    """Applies Newton-Schulz iteration to a single 2D matrix."""
    if x.ndim != 2:
        raise ValueError(f"_orthogonalize_2d_logic expects 2D input, got {x.shape}")

    if ns_coeffs.ndim > 2 or ns_coeffs.shape[-1] != 3:
        raise ValueError(
            "Newton-Schulz coefficients must have shape (3,) or (n, 3), "
            f"got {ns_coeffs.shape}"
        )

    def newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
        a = x @ x.T
        b = coeffs[1] * a + coeffs[2] * a @ a
        return coeffs[0] * x + b @ x

    transposed = False
    original_shape = x.shape
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True

    norm = jnp.linalg.norm(x)
    x /= norm + eps  # Ensure spectral norm is at most 1 (approximately)

    ns_coeffs = ns_coeffs.astype(x.dtype)
    if ns_coeffs.ndim == 1:

        def body_fn(_, current_x):
            return newton_schulz_iterator(current_x, ns_coeffs)

        x = jax.lax.fori_loop(0, ns_steps, body_fn, x)
    else:

        def scan_fn(current_x, abc):
            next_x = newton_schulz_iterator(current_x, abc)
            return next_x, None

        x, _ = jax.lax.scan(scan_fn, x, ns_coeffs)

    if transposed:
        x = x.T

    chex.assert_shape(x, original_shape)
    return x


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-8,
) -> jax.Array:
    r"""Orthogonalize via Newton-Schulz iteration. Handles 2D and 3D tensors.

    For 3D tensors (e.g., shape `(b, m, n)`), applies the orthogonalization
    independently to each `(m, n)` slice along the first dimension.

    Args:
        x: A matrix (2D) or a batch of matrices (3D) to orthogonalize.
        ns_coeffs: Coefficients for the Newton-schulz iterators.
            Must have shape (3,) or (n, 3) where n is the number of iterations.
        ns_steps: Number of Newton-schulz iterations.
            Ignored if `ns_coeffs` is a 2D array.
        eps: Term added to denominators to improve numerical stability.

    Returns:
        The orthogonalized matrix or batch of matrices.
    """
    if x.ndim == 2:
        return _orthogonalize_2d_logic(x, ns_coeffs, ns_steps, eps)
    elif x.ndim == 3:
        # Vmap over the first dimension for 3D tensors
        return jax.vmap(_orthogonalize_2d_logic, in_axes=(0, None, None, None))(
            x, ns_coeffs, ns_steps, eps
        )
    else:
        raise ValueError(f"Input must have shape (m, n) or (b, m, n), got {x.shape}")


class MuonState(NamedTuple):
    """State for the Muon algorithm."""

    count: chex.Array
    mu: base.Updates
    ns_coeffs: chex.Array


def scale_by_muon(
    ns_coeffs: Union[
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
    ] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
) -> base.GradientTransformation:
    r"""Rescale updates according to the Muon algorithm. Handles 2D/3D params.

    Args:
        ns_coeffs: Coefficients for the Newton-schulz method.
        ns_steps: Number of Newton-schulz iterations.
            Ignored if `ns_coeffs` is a tuple of tuples.
        beta: Decay rate for the exponentially weighted average of grads.
        eps: Term added to denominators to improve numerical stability.
        mu_dtype: Data type of the momentum accumulator.
        nesterov: Whether to use Nesterov momentum.
        adaptive: Whether to scale the updates by the dual norm of the
            original updates. See <https://arxiv.org/abs/2409.20325>.
            Handles 2D and 3D tensors appropriately.

    Returns:
        A `GradientTransformation` object.
    """
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = otu.tree_zeros_like(params, dtype=mu_dtype)
        ns_coeffs_ = jnp.asarray(ns_coeffs)
        if ns_coeffs_.ndim > 2 or ns_coeffs_.shape[-1] != 3:
            raise ValueError(
                f"ns_coeffs must have shape (3,) or (n, 3), got {ns_coeffs_.shape}"
            )
        return MuonState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            ns_coeffs=ns_coeffs_,
        )

    def update_fn(updates, state, params=None):
        del params
        mu = otu.tree_update_moment(updates, state.mu, beta, 1)
        count_inc = numerics.safe_increment(state.count)

        mu_hat_unbiased = otu.tree_bias_correction(mu, beta, count_inc)
        if nesterov:
            g_unbiased = otu.tree_bias_correction(updates, beta, count_inc)
            mu_hat = jax.tree.map(
                lambda m, g: beta * m + (1.0 - beta) * g, mu_hat_unbiased, g_unbiased
            )
        else:
            mu_hat = mu_hat_unbiased

        # Apply Newton-schulz orthogonalization (handles 2D and 3D).
        orthogonal_updates = jax.tree.map(
            lambda x: orthogonalize_via_newton_schulz(
                x, state.ns_coeffs, ns_steps, eps
            ),
            mu_hat,
        )

        if adaptive:
            # Adaptive scaling based on dual norm: scale = <mu_hat, orthogonal_updates>
            def adaptive_scale_leaf(mh_leaf, orth_leaf):
                if mh_leaf.ndim == 2:
                    scale = jnp.einsum(
                        "ij,ij->", mh_leaf, orth_leaf, optimize="optimal"
                    )
                    return scale * orth_leaf
                elif mh_leaf.ndim == 3:
                    scales = jnp.einsum(
                        "bij,bij->b", mh_leaf, orth_leaf, optimize="optimal"
                    )
                    # Expand dims of scales to (b, 1, 1) for broadcasting
                    return scales[:, None, None] * orth_leaf
                else:
                    # Pass through for non-matrix parameters
                    return orth_leaf

            updates = jax.tree.map(adaptive_scale_leaf, mu_hat, orthogonal_updates)
        else:
            updates = orthogonal_updates

        # Final aspect ratio scaling (applies to 2D and 3D)
        def aspect_ratio_scale(x):
            if x.ndim >= 2:  # Only apply to matrices
                aspect_ratio = x.shape[-1] / x.shape[-2]
                scale_factor = jnp.sqrt(jnp.maximum(1.0, aspect_ratio))
                return scale_factor * x
            else:
                return x  # Pass through for non-matrices

        updates = jax.tree.map(aspect_ratio_scale, updates)

        # Cast momentum for state update
        mu = otu.tree_cast(mu, mu_dtype)

        return updates, MuonState(
            count=count_inc,
            mu=mu,
            ns_coeffs=state.ns_coeffs,
        )

    return base.GradientTransformation(init_fn, update_fn)


def muon(
    learning_rate: base.ScalarOrSchedule,
    ns_coeffs: Union[
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
    ] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,  # Use separate eps for Adam
    adam_eps_root: float = 0.0,
    adam_weight_decay: float = 0.0,
) -> base.GradientTransformation:
    r"""Muon: Momentum Orthogonalized by Newton-schulz. Supports 2D/3D params.

    Applies Muon logic to parameters with `ndim == 2` or `ndim == 3`.
    For 3D parameters (shape `(b, m, n)`), the operations are applied
    independently to each `(m, n)` slice via `jax.vmap`.
    Other parameters (e.g., biases, norms - `ndim == 1` or `ndim == 0`)
    are updated using AdamW.

    Args:
        learning_rate: Global scaling factor or scheduler.
        ns_coeffs: Coefficients for the Newton-schulz method.
        ns_steps: Number of Newton-schulz iterations.
        beta: Decay rate for the Muon momentum.
        eps: Term added denominator in Muon orthogonalization.
        mu_dtype: Data type of the momentum accumulators.
        nesterov: Whether to use Nesterov momentum (applies to both).
        adaptive: Whether to scale Muon updates by dual norm.
        adam_b1: Exponential decay rate for Adam's first moment estimates.
        adam_b2: Exponential decay rate for Adam's second moment estimates.
        adam_eps: Epsilon term for Adam's normalization.
        adam_eps_root: Epsilon added to denominator before sqrt in Adam.
        adam_weight_decay: Weight decay factor for Adam.

    Returns:
        The corresponding `GradientTransformation`.
    """

    def is_muon_param(param: chex.Array) -> bool:
        # Apply Muon to 2D and 3D tensors
        return param.ndim in (2, 3)

    param_labels: Callable = lambda params: jax.tree.map(
        lambda x: "muon" if is_muon_param(x) else "adam", params
    )

    return combine.partition(
        transforms={
            "muon": combine.chain(
                scale_by_muon(
                    ns_coeffs=ns_coeffs,
                    ns_steps=ns_steps,
                    beta=beta,
                    eps=eps,
                    mu_dtype=mu_dtype,
                    nesterov=nesterov,
                    adaptive=adaptive,
                ),
                transform.scale_by_learning_rate(learning_rate),
            ),
            "adam": alias.adamw(
                learning_rate=learning_rate,
                b1=adam_b1,
                b2=adam_b2,
                eps=adam_eps,
                eps_root=adam_eps_root,
                weight_decay=adam_weight_decay,
                mu_dtype=mu_dtype,
                nesterov=nesterov,
            ),
        },
        param_labels=param_labels,
    )
