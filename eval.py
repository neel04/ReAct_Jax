import argparse
import os
from typing import Any, List

import equinox as eqx
import jax
import jax.numpy as jnp
import lm_eval
from jaxtyping import Array, PRNGKeyArray
from lm_eval.api.model import TemplateLM
from lm_eval.tasks import TaskManager
from tqdm import tqdm

from ReAct.data.owt import OpenWebTextDataset as OWT
from ReAct.model.baseline import GPT
from ReAct.model.react import React
from ReAct.utils.arg_parser import get_evaluation_args
from ReAct.utils.helpers import load_eqx_obj
from ReAct.utils.logger import UnifiedLogger
from ReAct.utils.sharding import get_strategy


class Evaluator:
    def __init__(self, args: argparse.Namespace, key: PRNGKeyArray):
        self.pad_token = 50257
        self.key = key
        self.args = args

        dummy_dataset = OWT(split="train")

        self.decode_fn = dummy_dataset.tok.decode
        self.encode_fn = dummy_dataset.tok.encode

        self.strategy = get_strategy(self.args.strategy)

    def skeleton_model(self, is_baseline: bool) -> GPT | React:
        if not is_baseline:
            model = React(
                n_heads=self.args.n_heads,
                seqlen=self.args.seqlen,
                max_iters=self.args.max_iters,
                num_blocks=self.args.num_blocks,
                width=self.args.width,
                drop_rate=0.0,
                vocab_size=self.args.num_classes,
                key=self.key,
                strategy=self.strategy,
            )
        else:
            model = GPT(
                n_heads=self.args.n_heads,
                seqlen=self.args.seqlen,
                num_blocks=self.args.num_blocks,
                width=self.args.width,
                drop_rate=0.0,
                vocab_size=self.args.num_classes,
                key=self.key,
                strategy=self.strategy,
            )

        return model

    def encode_input(self, my_input: str, obey_maxlen: bool = True) -> Array:
        encoded = self.encode_fn(my_input, obey_maxlen=obey_maxlen)["input_ids"]
        encoded = jnp.asarray([i for i in encoded if i != self.pad_token])

        return encoded

    def run_lm_evaluation(self):
        model = self.skeleton_model(self.args.baseline)
        model = load_eqx_obj(
            self.args.checkpoint_path,
            model if self.args.baseline else eqx.filter(model, eqx.is_array),
        )

        class MyLM(TemplateLM):
            def __init__(
                self,
                model: eqx.Module,
                encode_fn: Any,
                decode_fn: Any,
                args: argparse.Namespace,
            ):
                super().__init__()

                self.model = eqx.nn.inference_mode(model)
                self.encode_fn = encode_fn
                self.decode_fn = decode_fn
                self.args = args

            @property
            def eot_token_id(self) -> Any:
                return 50304

            def tok_encode(self, string: str, **kwargs) -> list[int]:
                encoded: Array = self.encode_fn(string, obey_maxlen=False)
                encoded = jnp.asarray([i for i in encoded if i != self.eot_token_id])

                return encoded.tolist()

            def _calc_ll(
                self, seq: Array, lengths: tuple[int, int], target: Array
            ) -> Array:
                arrlen, tgtlen = lengths
                pad_mask = jnp.where(seq == self.eot_token_id, 0, 1)
                key = jax.random.PRNGKey(0)

                def fwd(seq: Array, pad_mask: Array, key: PRNGKeyArray) -> Array:
                    if self.args.baseline:
                        logits = self.model(seq, pad_mask, False, key)
                    else:
                        logits = self.model(
                            seq,
                            self.args.max_iters,
                            jnp.ones_like(seq),
                            False,
                            False,
                            key,
                        )[0]

                    return jax.nn.log_softmax(logits, axis=-1)

                probs = fwd(seq, pad_mask, key)

                target_log_probs = (
                    probs[jnp.arange(arrlen, arrlen + tgtlen), target[:tgtlen]]
                    * pad_mask[:tgtlen]
                )

                return target_log_probs.sum()

            def _loglikelihood_tokens(
                self, requests: List, **kwargs
            ) -> list[tuple[float, bool]]:
                output = []

                for request in tqdm(requests):
                    context, target = request[-2], request[-1]

                    arr, target = (
                        jnp.asarray(context).astype(int),
                        jnp.asarray(target).astype(int),
                    )
                    arrlen, tgtlen = len(arr), len(target)

                    seq = jnp.concat([arr, target])[-self.args.seqlen :]
                    seq = jnp.pad(
                        seq,
                        (0, self.args.seqlen - seq.shape[0]),
                        constant_values=self.eot_token_id,
                    )

                    ll = self._calc_ll(seq, (arrlen, tgtlen), target)
                    output.append((ll.item(), 1))

                return output

            def loglikelihood_rolling(
                self, requests, disable_tqdm: bool = False
            ) -> list[float]:
                """
                Compute rolling log-likelihood for each request by:
                1. Breaking input into appropriate chunks based on max context length
                2. Computing log-likelihood for each chunk with maximum possible context
                3. Ensuring each token is predicted exactly once

                Args:
                    requests: List of request tuples containing (context,) strings
                    disable_tqdm: Whether to disable progress bar

                Returns:
                    List of log-likelihood scores for each request
                """
                output = []

                for request in tqdm(requests, disable=disable_tqdm):
                    context = request.arguments[0]

                    # Encode full context
                    tokens = []

                    for chunk in range((len(context) // 4096) + 1):
                        tokens.extend(
                            self.tok_encode(context[chunk * 4096 : (chunk + 1) * 4096])
                        )

                    tokens = jnp.asarray(tokens)

                    # For longer contexts, process in chunks with maximum context
                    total_ll = 0.0
                    chunk_size = self.args.seqlen

                    # Process full chunks first
                    for i in range(len(tokens) // chunk_size + 1):
                        # If context fits in one window, process it directly
                        chunk = tokens[i * chunk_size : (i + 1) * chunk_size + 1]

                        if len(chunk) < self.args.seqlen:
                            # Pad sequence to model's expected length
                            seq = jnp.pad(
                                chunk,
                                (0, self.args.seqlen - len(chunk) + 1),
                                constant_values=self.eot_token_id,
                            )
                            # Calculate log-likelihood for the whole sequence
                            ll = self._calc_ll(
                                seq[:-1],
                                (0, len(chunk)),
                                jnp.roll(seq, -1)[:-1],
                            )

                            total_ll += ll.item()
                        else:
                            ll = self._calc_ll(
                                chunk[:-1],
                                (0, chunk_size),
                                jnp.roll(chunk, -1)[:-1],
                            )

                            total_ll += ll.item()

                    output.append(total_ll)

                return output

            def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
                raise NotImplementedError

        lm_obj = MyLM(
            model=model,
            encode_fn=self.encode_input,
            decode_fn=self.decode_fn,
            args=self.args,
        )

        task_manager = TaskManager()

        results = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=[self.args.task],
            num_fewshot=None,
            task_manager=task_manager,
        )

        return results['results'] # type: ignore


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    args = get_evaluation_args()
    logger = UnifiedLogger(args, level="DEBUG")
    my_logger = logger.my_logger()

    my_logger.warning(
        "Make sure to provide the correct args per the model configuration - as it cant be autodetected!"
    )
    my_logger.warning("These are: max_iters| baseline | num_blocks | width | n_heads")
    print(f"{'-'*50}\n")

    assert args.checkpoint_path is not None, "Please provide a checkpoint path"
    assert os.path.exists(
        args.checkpoint_path
    ), "Please provide a valid checkpoint path | File does not exist"

    evaluator = Evaluator(args, key)

    # Run LM evaluation
    eval_results = evaluator.run_lm_evaluation()
    print("\nLM Evaluation Results:")
    print(eval_results)
