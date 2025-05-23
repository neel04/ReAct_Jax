import math
import os
from logging import Logger
from typing import Any, Callable, List, Optional, Tuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax_array_info import sharding_info
from jaxtyping import Array, PRNGKeyArray, PyTree

import wandb

T = TypeVar('T')

class Profiler:
    def __init__(
        self, activate_profiler: bool = True, logdir: str = "./profiles/"
    ) -> None:
        self.warmup_steps = 50
        self.activate_profiler = activate_profiler
        self.logdir = logdir

    def start_prof(self, step: int) -> None:
        if step == self.warmup_steps:
            if self.activate_profiler:
                print(f'Started TensorBoard Profiler at: {self.logdir}')
                jax.profiler.start_trace(self.logdir)

    def stop_prof(self, output: Array, step: int) -> Array:
        if step == self.warmup_steps:
            if self.activate_profiler:
                output = output.block_until_ready() # wait for output
                jax.profiler.stop_trace()
                print(f'Stopped Profiler at: {self.logdir}')

            self.activate_profiler = False

        return output

def convert_flops(params: int) -> str:
    if params == 0:
        return "0"

    size_name = (
        "",
        "KFLOPs",
        "MFLOPs",
        "GFLOPs",
        "TFLOPs",
        "PFLOPs",
        "EFLOPs",
        "ZFLOPs",
        "YFLOPs",
    )
    i = int(math.floor(math.log(params, 1000)))
    p = math.pow(1000, i)
    s = round(params / p, 2)

    return "%s %s" % (s, size_name[i])


def calc_performance_metrics(args, my_logger: Logger) -> None:
    """
    Estimates FLOPs consumed during a single fwd + bwd pass.
    Taken from EleutherAI's GPT-NeoX repo: https://rb.gy/33d6zg

    Returns: the total number of FLOPs
    """
    iter_factor = 3
    args.tokens = args.batch_size * args.seqlen
    args.kv_size_ratio = 1

    my_logger.warning("! Ignoring activation checkpointing in FLOPs calculation !")

    qkv_flops = int(
        iter_factor
        * 2
        * (1 + 2 * args.kv_size_ratio)
        * args.num_blocks
        * args.tokens
        * args.width
        * args.width
    )
    attention_matrix_flops = (
        iter_factor * 2 * args.num_blocks * args.tokens * args.seqlen * args.width
    )
    attention_over_values_flops = (
        iter_factor * 2 * args.num_blocks * args.tokens * args.seqlen * args.width
    )
    linear_projection_flops = (
        iter_factor * 2 * args.num_blocks * args.tokens * args.width * args.width
    )
    ffn_flops = (
        iter_factor * 16 * args.num_blocks * args.tokens * args.width * args.width
    )

    # handle NewGELU
    ffn_flops *= 3.75

    embedding_flops = 6 * args.tokens * args.width * args.num_classes
    total_flops = (
        qkv_flops
        + attention_matrix_flops
        + attention_over_values_flops
        + linear_projection_flops
        + ffn_flops
        + embedding_flops
    )
    my_logger.info(
        f"Total FLOPs for the Model: {convert_flops(total_flops)} for a single fwd + bwd pass\n"
    )


def xla_calc_flops(fn: Callable, static_argnums: Tuple[int], args: Tuple[Any], my_logger: Logger) -> int:
    """
    Estimates FLOPs consumed during `fn` execution.
    Use's XLA HLO analysis to estimate FLOPs.

    Returns: the total number of FLOPs
    """
    compiled = jax.jit(fn, static_argnums=static_argnums).lower(*args).compile()
    flops = compiled.cost_analysis()[0]["flops"] # type: ignore
    my_logger.info(f"XLA estimate of Total FLOPs for {fn.__name__}: {convert_flops(int(flops))}\n")

    return flops


def safe_softmax(x: Array, axis: int) -> Array:
    """
    Perform softmax on the input array `x` with numerical stability
    (full precision softmax)
    """
    dtype = jnp.result_type(x.dtype, jnp.float32)
    return jax.nn.softmax(x.astype(dtype), axis=axis).astype(x.dtype)


def half_precision(model: eqx.Module) -> eqx.Module:
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x, model
    )

def viz_obj(model: PyTree):
    model = eqx.filter(model, eqx.is_array)

    def viz_fn(leaf):
        print(f"\n=== leaf: {leaf.shape} ===\n")
        return sharding_info(leaf)

    jax.tree_util.tree_map(viz_fn, model)

def get_spec_on_larger_dim(leaf: PyTree, key: str = "model") -> List[str | None]:
    p_spec = [
        key if i == leaf.shape.index(max(leaf.shape)) else None
        for i in range(len(leaf.shape))
    ]

    return p_spec


def megatron_init(weight: Array, key: PRNGKeyArray) -> Array:
    """
    Init all the weights with the Megatron paper init
    """
    dims = weight.shape
    stddev = (0.33 / dims[0]) ** 0.5
    lim = 1 / math.sqrt(dims[1])

    return jax.random.uniform(key, dims, minval=-lim, maxval=lim) * stddev

def zero_init(weight: Array) -> Array:
    """
    Init all the weights with zeroes.
    """
    return jnp.zeros_like(weight, dtype=weight.dtype)

def get_weights(m: PyTree, layer: PyTree):

    def is_linear(x: Any):
        return isinstance(x, eqx.nn.Linear) or isinstance(x, layer)

    return [
        x.weight
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]

@eqx.filter_jit
def get_hist(tree: PyTree, num_bins: int = 64) -> Any:
    """
    Compute histogram, handling for NaNs safely.
    Returns: Tuple[Array, Array] but wandbs typehinting covereage is so ass.
    """
    leaves = get_leaves(tree)
    
    return jnp.histogram(
        leaves, bins=num_bins, range=(jnp.nanmin(leaves), jnp.nanmax(leaves))
    )


def save_eqx_obj(save_dir: str, filename: str, obj: tuple):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    eqx.tree_serialise_leaves(filename, obj)

def get_leaves(x: T) -> T:
    return jax.flatten_util.ravel_pytree(
        jax.tree_util.tree_flatten(x, eqx.is_array)[0]
    )[0]

def load_eqx_obj(filepath: str, obj: PyTree[Any]) -> PyTree[Any]:
    return eqx.tree_deserialise_leaves(path_or_file=filepath, like=obj)


def broad_to_bsz(arr: Array, shape: tuple) -> Array:
    return jnp.broadcast_to(arr, shape)


def count_params(model: eqx.Module) -> None:
    def params_fn(model: PyTree):
        return sum(
            x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
        )

    num_params, non_embed_params = (
        params_fn(model),
        params_fn(model.main_block),
    )

    unshared_params = 0
    num_params /= 1_000_000
    non_embed_params /= 1_000_000

    if hasattr(model.main_block, "unshared_layers"):
        unshared_params += params_fn(model.main_block.unshared_layers) / 1_000_000

    if hasattr((layers := model.main_block.attention_layers)[0], "unshared_layers"):
        unshared_params += (params_fn(layers[0].unshared_layers) / 1_000_000) * len(layers)

    if hasattr(model, "unshared_layers"):
        unshared_params += params_fn(model.unshared_layers) / 1_000_000

    print(f"\nUnshared Parameters: {unshared_params}M")
    print(
        f"Model # of parameters: {num_params:.2f}M\n# of recurrent parameters: {non_embed_params:.2f}M\n"
    )


def get_rand_nums(
    key: PRNGKeyArray,
    lower_bound: int,
    upper_bound: int,
    bsz: int,
    bias_val: Optional[int] = None,
) -> Array:
    """
    Generate random numbers from a uniform distribution
    or bias it towards a certain value, if provided
    """
    if bias_val is None:
        dist = jax.random.randint(
            key, shape=(bsz,), minval=lower_bound, maxval=upper_bound
        )
    else:
        dist = jnp.clip(
            jax.random.normal(key, (bsz,)) * (bias_val**0.5) + bias_val + 1,
            lower_bound,
            upper_bound,
        )

    return dist.astype(int)

def download_artifact(artifact_path: str):
    api = wandb.Api()

    if api.artifact_exists(artifact_path):
        artifact = api.artifact(artifact_path)
        datadir = artifact.download(root="./", skip_cache=True)
        print(f"\nArtifact downloaded at {datadir}. Ensure this chkp is loaded.")
    else:
        print(f"Warning: Artifact {artifact_path} does not exist.\n")
