from argparse import Namespace
from dataclasses import dataclass
from typing import Type, TypeVar


@dataclass
class TrainingArgs:
    """Training arguments for the model."""

    num_blocks: int
    strategy: str
    model_axis: int
    dataset: str
    width: int
    rank: int
    drop_rate: float
    num_classes: int
    n_heads: int
    max_iters: int
    seqlen: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps: int
    grad_clip: float
    optimizer_type: str
    epochs: int
    beta_1: float
    beta_2: float
    nesterov: bool
    save_interval: int
    log_interval: int
    save_dir: str
    debug: bool
    bf16: bool
    exp_logging: bool
    group: str
    checkpoint_path: str | None
    resume: bool | str
    baseline: bool
    tune_hyperparams: bool
    sweep_metadata: str
    accum_steps: int
    profile: bool


@dataclass
class InferenceArgs:
    """Inference arguments for the model."""

    seqlen: int
    max_iters: int
    num_classes: int
    num_tokens: int
    checkpoint_path: str
    n_heads: int
    width: int
    baseline: bool
    num_blocks: int
    system_prompt: str
    prompt: str
    temp: float
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float


@dataclass
class EvaluationArgs:
    """Evaluation arguments for the model."""

    seqlen: int
    max_iters: int
    num_classes: int
    num_tokens: int
    checkpoint_path: str
    n_heads: int
    width: int
    baseline: bool
    num_blocks: int
    strategy: str
    task: str


Args = TypeVar("Args", TrainingArgs, EvaluationArgs, InferenceArgs)


def from_namespace(namespace: Namespace, dataclass_type: Type[Args]) -> Args:
    """Convert an argparse Namespace to a dataclass.

    Args:
        namespace: The argparse Namespace object
        dataclass_type: The target dataclass type

    Returns:
        An instance of the dataclass populated with values from the namespace
    """
    # Convert namespace to dict and initialize the dataclass
    try:
        return dataclass_type(**vars(namespace))
    except TypeError as e:
        raise ValueError(
            f"Provided args didn't override entire struct. Check the integrity of the structs and the argparser: {e}"
        ) from e
