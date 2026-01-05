import math
import os
from contextlib import contextmanager
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import regex as re
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from jax.experimental import multihost_utils
from jax_array_info import sharding_info
from jaxtyping import Array, PRNGKeyArray, PyTree
from torch.utils.data import DataLoader as TorchDataLoader

import wandb

T = TypeVar('T')

@contextmanager
def temp_cwd(path: Path):
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)

class Profiler:
    def __init__(
        self, activate_profiler: bool = True, logdir: str = "./profiles/"
    ) -> None:
        self.options = jax.profiler.ProfileOptions()
        self.options.host_tracer_level = 2
        self.options.device_tracer_level = 1
        self.options.python_tracer_level = 1
        self.options.advanced_configuration = {
            "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC",
        }

        self.warmup_steps = 100
        self.activate_profiler = activate_profiler
        self.logdir = logdir

    def start_prof(self, step: int) -> None:
        if step == self.warmup_steps:
            if self.activate_profiler:
                print(f"Started TensorBoard Profiler at: {self.logdir}")
                jax.profiler.start_trace(
                    self.logdir,
                    create_perfetto_link=True,
                    create_perfetto_trace=True,
                    profiler_options=self.options,
                )

    def stop_prof(self, w_logger: Any, output: Array, step: int) -> Array:
        if step == self.warmup_steps:
            if self.activate_profiler:
                output = output.block_until_ready()  # wait for output
                jax.profiler.stop_trace()
                print(f"Stopped Profiler at: {self.logdir}")
                self.upload_to_wandb(w_logger)

            self.activate_profiler = False

        return output

    def upload_to_wandb(self, w_logger: Any):
        print("Uploading to W&B...")
        artifact = wandb.Artifact('tb_profile', type='profile')
        artifact.add_dir("profiles/")
        w_logger.log_artifact(artifact)

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
def get_hist(key: PRNGKeyArray, tree: PyTree, num_bins: int = 64) -> Any:
    """
    Compute histogram, handling for NaNs safely.
    Returns: Tuple[Array, Array] but wandbs typehinting covereage is so ass.
    """
    leaves = jax.random.choice(key, get_leaves(tree), (8192,), False)

    return jnp.histogram(
        leaves, bins=num_bins, range=(jnp.nanmin(leaves), jnp.nanmax(leaves))
    )


@eqx.filter_jit
def chunked_histogram(key: PRNGKeyArray, tree: PyTree, num_bins: int = 64):
    """
    Compute histogram in a blockwise fashion.
    """
    # 1. Global Pass: Find global Min/Max
    # Map each leaf to its scalar min/max
    mins = jax.tree.map(jnp.nanmin, tree)
    maxs = jax.tree.map(jnp.nanmax, tree)

    # Reduce scalar leaves to global scalars
    g_min = jax.tree_util.tree_reduce(jnp.fmin, mins, initializer=jnp.inf)
    g_max = jax.tree_util.tree_reduce(jnp.fmax, maxs, initializer=-jnp.inf)

    # 2. Local Pass: Compute histogram for each leaf using GLOBAL range
    def leaf_hist(leaf: PyTree):
        counts, _ = jnp.histogram(leaf, bins=num_bins, range=(g_min, g_max))
        return counts.astype(jnp.float32)

    # Map to get counts per leaf, then reduce (sum) them
    leaf_counts = jax.tree.map(leaf_hist, tree)
    total_counts = jax.tree_util.tree_reduce(jnp.add, leaf_counts)

    # Recreate bin edges (cheap linear space)
    bin_edges = jnp.linspace(g_min, g_max, num_bins + 1)

    return total_counts, bin_edges

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

    if hasattr(model.main_block, "unshared_layers"):
        unshared_params += params_fn(model.main_block.unshared_layers)

    if hasattr(model.main_block, "attention_layers"):
        if hasattr(model.main_block.attention_layers, "unshared_layers"):
            unshared_params += params_fn(model.main_block.attention_layers.unshared_layers)

    if hasattr(model, "unshared_layers"):
        unshared_params += params_fn(model.unshared_layers)

    num_params /= 1_000_000
    non_embed_params /= 1_000_000
    unshared_params /= 1_000_000

    print(f"\nUnshared Parameters: {unshared_params:.2f}M")
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

def download_artifact(artifact_path: str, chkp_type: str = "OptunaCheckpoint", save_dir: str = "./") -> bool:
    api = wandb.Api()

    if api.artifact_exists(artifact_path):
        print("Downloading artifact...")
        artifact = api.artifact(artifact_path, chkp_type)
        datadir = artifact.download(root=save_dir, skip_cache=True)
        print(f"\nArtifact downloaded at {datadir}. Ensure this chkp is loaded.")
        return True

    print(f"Warning: Artifact {artifact_path} does not exist.\n")
    return False

def fetch_resume_progress(resume: bool | str, save_dir: str, chkp_type: str) -> tuple[int, int]:
    """
    Downloads the latest checkpoint artifact (if a resume string is provided)
    and extracts the latest epoch and step from files in `save_dir`.

    The `resume` string may be of the form:
      - "<run_id>" or "entity/project/<run_id>"
      - "entity/project/<run_id> + <epoch> + <step>"

    Returns:
      (step, epoch) as integers. Defaults to (0, 0) if nothing found.
    """
    if not isinstance(resume, str):
        return 0, 0

    # Extract the run path/name before any optional + epoch/step suffixes
    prefix = resume.split("+")[0].strip()

    if len(prefix) == 0:
        return 0, 0

    # Build full artifact path
    artifact_path = (
        f"{prefix}:latest"
        if prefix.count("/") >= 2
        else f"neel/ReAct_Jax/{prefix}:latest"
    )

    # Best-effort download; ignore failures and fall back to parsing numbers
    _ = download_artifact(artifact_path, save_dir=save_dir, chkp_type=chkp_type)

    # Inspect local directory for any .eqx files and pick the latest by (epoch, step)
    try:
        files = [
            os.path.join(save_dir, file)
            for file in os.listdir(save_dir)
            if file.endswith("eqx")
        ]

        if len(files) > 0:
            epoch, step = max(
                [re.findall(r"\d+", file) for file in files],
                key=lambda x: (int(x[0]), int(x[1])),
            )
            return int(step), int(epoch)
    except FileNotFoundError:
        pass

    # Fallback: try to parse epoch and step from the resume string if provided
    nums = [int(x.strip()) for x in resume.split("+")[1:] if x.strip().isdigit()]
    if len(nums) >= 2:
        epoch, step = nums[0], nums[1]
        return int(step), int(epoch)

    return 0, 0

class IterableDatasetWithLen(IterableDataset):
    def __init__(
        self,
        dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict,
        length: int,
    ):
        self.dataset = dataset
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)

    def __iter__(self):
        for item in self.dataset:
            yield item

def _build_torch_prefetch_loader(
    loader: Dataset
    | DatasetDict
    | IterableDataset
    | IterableDatasetDict
    | IterableDatasetWithLen,
    prefetch_size: int,
) -> TorchDataLoader:
    core_count = 32 if os.cpu_count() >= 32 else 0  # type: ignore
    prefetch_size: int = None if core_count == 0 else prefetch_size  # type: ignore

    print(f"Using {core_count} cores for the dataloader!")

    return TorchDataLoader(
        loader,  # pyright: ignore[reportArgumentType]
        batch_size=1,
        num_workers=0,
        # prefetch_factor=prefetch_size,
        # persistent_workers=True if core_count > 0 else False,
        pin_memory=False,
    )

def block_prefetcher(loader: Iterator[Any], prefetch_size: int = 512):
    sentinel = object()

    def fill_buffer():
        buffer = []

        for _ in range(prefetch_size):
            try:
                buffer.append(next(loader))
            except StopIteration:
                break

        buffer.append(sentinel)
        return buffer

    buffer = fill_buffer()

    while True:
        for item in buffer:
            if item is sentinel:
                break
            yield item

        if len(buffer) <= 1:
            break

        buffer = fill_buffer()

def broadcast_batch(
    loader: Dataset
    | DatasetDict
    | IterableDataset
    | IterableDatasetDict
    | IterableDatasetWithLen,
    batch_size: int,
    seqlen: int,
    prefetch_size: int = 128,
) -> Iterator[dict[str, tuple[Array, Array, Array]]]:
    """
    Ensures only process 0 touches the real loader while all hosts receive
    identical arrays via `broadcast_one_to_all`. Optionally prefetches
    `prefetch_size` batches ahead on the primary host. When possible, a
    multi-worker torch DataLoader sustains throughput on the primary host.
    """
    is_primary = jax.process_index() == 0

    if is_primary:
        iterator: Iterator[Any] = block_prefetcher(iter(loader)) # FIX: Bypasses torch dataloader
    else:
        iterator = iter(range(len(loader)))  # pyright: ignore[reportArgumentType]

    zero_batch = jnp.zeros((batch_size, seqlen), dtype=jnp.int32)

    for batch in iterator:
        if is_primary:
            seq, label, pad_mask = jnp.asarray(batch["text"])  # type: ignore[index]
        else:
            seq = label = pad_mask = zero_batch

        seq, label, pad_mask = multihost_utils.broadcast_one_to_all(
            (seq, label, pad_mask), is_source=is_primary

        )

        seq, label, pad_mask = jax.tree_util.tree_map(
            lambda x: x.squeeze(), (seq, label, pad_mask)
        )

        yield {"text": (seq, label, pad_mask)}
