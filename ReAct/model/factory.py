from typing import Literal, TypeAlias

from jaxtyping import PRNGKeyArray

from ReAct.model.baseline import GPT
from ReAct.model.naive_ut import React as NaiveReact
from ReAct.model.react import React
from ReAct.utils.arg_types import EvaluationArgs, InferenceArgs, TrainingArgs
from ReAct.utils.sharding import Sharding

ModelArgs: TypeAlias = TrainingArgs | InferenceArgs | EvaluationArgs
ModelKind: TypeAlias = Literal["baseline", "react", "naive"]
UTModel: TypeAlias = React | NaiveReact
Model: TypeAlias = GPT | UTModel


def resolve_model_kind(*, baseline: bool, naive: bool) -> ModelKind:
    assert not (baseline and naive), (
        "Model flags are mutually exclusive: choose --baseline or --naive, not both."
    )

    if baseline:
        return "baseline"
    if naive:
        return "naive"
    return "react"


def build_model(
    *,
    args: ModelArgs,
    key: PRNGKeyArray,
    strategy: Sharding,
    drop_rate: float | None = None,
) -> Model:
    resolved_drop_rate = (
        drop_rate if drop_rate is not None else getattr(args, "drop_rate", 0.0)
    )
    model_kind = resolve_model_kind(baseline=args.baseline, naive=args.naive)

    match model_kind:
        case "baseline":
            return GPT(
                n_heads=args.n_heads,
                seqlen=args.seqlen,
                num_blocks=args.num_blocks,
                width=args.width,
                drop_rate=resolved_drop_rate,
                vocab_size=args.num_classes,
                key=key,
                strategy=strategy,
            )
        case "naive":
            return NaiveReact(
                rank=args.rank,
                n_heads=args.n_heads,
                seqlen=args.seqlen,
                max_iters=args.max_iters,
                num_blocks=args.num_blocks,
                width=args.width,
                drop_rate=resolved_drop_rate,
                vocab_size=args.num_classes,
                key=key,
                strategy=strategy,
            )
        case "react":
            return React(
                rank=args.rank,
                n_heads=args.n_heads,
                seqlen=args.seqlen,
                max_iters=args.max_iters,
                num_blocks=args.num_blocks,
                width=args.width,
                drop_rate=resolved_drop_rate,
                vocab_size=args.num_classes,
                key=key,
                strategy=strategy,
            )
        case _:
            raise AssertionError(f"Unhandled model kind: {model_kind}")
