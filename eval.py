import argparse
import os
from typing import Any 

import equinox as eqx
import jax
import jax.numpy as jnp
import lm_eval
from jaxtyping import Array, PRNGKeyArray
from lm_eval.api.model import TemplateLM
from lm_eval.tasks import TaskManager

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

    def encode_input(self, my_input: str) -> Array:
        encoded = self.encode_fn(my_input)["input_ids"]
        encoded = jnp.asarray([i for i in encoded if i != self.pad_token])

        return encoded

    def run_lm_evaluation(self):
        model = self.skeleton_model(self.args.baseline)
        model = load_eqx_obj(self.args.checkpoint_path, model)

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
                encoded: Array = self.encode_fn(string)
                encoded = jnp.asarray([i for i in encoded if i != self.eot_token_id])

                return encoded.tolist()

            def _calc_ll(self, arr: Array, target: Array, **kwargs) -> Array:
                arr, target = jnp.asarray(arr), jnp.asarray(target)

                seq = jnp.concat([arr, target])
                seq = jnp.pad(
                    seq,
                    (0, self.args.seqlen - seq.shape[0]),
                    constant_values=self.eot_token_id,
                )

                pad_mask = jnp.where(seq == self.eot_token_id, 0, 1)
                key = jax.random.PRNGKey(0)

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

                probs = jax.nn.log_softmax(logits, axis=-1)

                breakpoint()
                # select logprobs for the target token
                target_log_probs = jnp.take_along_axis(
                    probs, target[:, :, None], axis=-1
                ).squeeze(-1)

            def _loglikelihood_tokens(self, requests: list, **kwargs) -> list[tuple[float, bool]]:
                breakpoint()
                for request in requests:
                    context, target = request[-2], request[-1]
                    self._calc_ll(context, target, kwargs=kwargs)

                ...


            def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
                return super().loglikelihood_rolling(requests, disable_tqdm)

            def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
                raise NotImplementedError 

                return super().generate_until(requests, disable_tqdm)
            
            '''
            def _loglikelihood(self, requests: list[Instance]):
                res = []

                for request in requests:
                    context, continuation = request.args

                    inp = self.encode_fn(context + continuation)
                    context_enc = self.encode_fn(context)
                    cont_enc = inp[len(context_enc) :]

                    if len(inp) > self.args.seqlen:
                        inp = inp[-self.args.seqlen :]
                        context_enc = context_enc[-(self.args.seqlen - len(cont_enc)) :]

                    pad_amount = self.args.seqlen - len(inp)
                    inp = jnp.pad(
                        jnp.array(inp),
                        (0, pad_amount),
                        constant_values=self.args.pad_token,
                    )

                    key = jax.random.PRNGKey(0)
                    pad_mask = jnp.where(inp == self.args.pad_token, 0, 1)

                    if self.args.baseline:
                        logits = self.model(
                            inp[None, :], pad_mask[None, :], False, key
                        )[0, : len(context_enc) + len(cont_enc) - 1]
                    else:
                        logits = self.model(
                            inp[None, :],
                            self.args.max_iters,
                            pad_mask[None, :],
                            False,
                            False,
                            key,
                        )[0][0, : len(context_enc) + len(cont_enc) - 1]

                    log_probs = jax.nn.log_softmax(logits, axis=-1)
                    cont_log_probs = (
                        log_probs[len(context_enc) - 1 :]
                        .at[jnp.arange(len(cont_enc)), jnp.array(cont_enc)]
                        .get()
                    )

                    total_log_prob = cont_log_probs.sum()
                    res.append((total_log_prob.item(), True))
                return res

            def loglikelihood_rolling(self, requests):
                res = []
                for context, continuation in requests:
                    inp = self.encode_fn(context + continuation)
                    total_log_prob = 0.0

                    for i in range(len(context), len(inp)):
                        window = inp[max(0, i - self.args.seqlen + 1) : i + 1]
                        pad_amount = self.args.seqlen - len(window)
                        window = jnp.pad(
                            jnp.array(window),
                            (pad_amount, 0),
                            constant_values=self.args.pad_token,
                        )

                        key = jax.random.PRNGKey(0)
                        pad_mask = jnp.where(window == self.args.pad_token, 0, 1)
                        if self.args.baseline:
                            logits = self.model(
                                window[None, :], pad_mask[None, :], False, key
                            )[0, -1]
                        else:
                            logits = self.model(
                                window[None, :],
                                self.args.max_iters,
                                pad_mask[None, :],
                                False,
                                False,
                                key,
                            )[0][0, -1]
                        log_probs = jax.nn.log_softmax(logits, axis=-1)
                        total_log_prob += log_probs[inp[i]].item()

                    res.append((total_log_prob, True))
                return res

            def generate_until(self, requests):
                res = []
                for context, until in requests:
                    inp = self.encode_fn(context)
                    generated = inp.copy()

                    while not any(self.decode_fn(generated).endswith(u) for u in until):
                        if len(generated) >= self.args.seqlen:
                            break

                        window = generated[-self.args.seqlen :]
                        pad_amount = self.args.seqlen - len(window)
                        window = jnp.pad(
                            jnp.array(window),
                            (pad_amount, 0),
                            constant_values=self.args.pad_token,
                        )

                        key = jax.random.PRNGKey(0)
                        pad_mask = jnp.where(window == self.args.pad_token, 0, 1)
                        if self.args.baseline:
                            logits = self.model(
                                window[None, :], pad_mask[None, :], False, key
                            )[0, -1]
                        else:
                            logits = self.model(
                                window[None, :],
                                self.args.max_iters,
                                pad_mask[None, :],
                                False,
                                False,
                                key,
                            )[0][0, -1]
                        next_token = jax.random.categorical(key, logits).item()
                        generated.append(next_token)

                    res.append(self.decode_fn(generated[len(inp) :]))
                return res
            '''

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

        return results


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
