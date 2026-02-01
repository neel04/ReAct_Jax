import argparse
import os

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ReAct.data.tokenizer import Tok
from ReAct.model.factory import Model, build_model
from ReAct.utils.arg_parser import get_inference_args
from ReAct.utils.arg_types import Args
from ReAct.utils.helpers import count_params, load_eqx_obj
from ReAct.utils.logger import UnifiedLogger
from ReAct.utils.sharding import get_strategy


class Inferencer:
    def __init__(self, args: Args, key: PRNGKeyArray):
        self.pad_token = 50257
        self.key = key
        self.args = args

        tok = Tok(vocab_dir=None, max_length=512)

        self.decode_fn = tok.decode
        self.encode_fn = tok.encode

        self.strategy = get_strategy('ddp')

    def skeleton_model(self) -> Model:
        return build_model(
            args=self.args,
            key=self.key,
            strategy=self.strategy,
            drop_rate=0.0,
        )

    def encode_input(self, my_input: str) -> Array:
        encoded = self.encode_fn(my_input)['input_ids']
        encoded = jnp.asarray([i for i in encoded if i != self.pad_token]).astype(int)

        return encoded

    def sample_model(
        self,
        model: eqx.Module,
        my_input: str,
        num_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        max_repeat_tokens: int = 3,
    ) -> str:
        '''
        Samples the model autoregressively using more advanced techniques:
        - Temperature scaling
        - Top-k sampling
        - Nucleus (top-p) sampling
        - Repetition penalty
        '''
        inference_model = eqx.nn.inference_mode(model)

        if hasattr(self.args, "system_prompt"):
            prompt = self.args.system_prompt + my_input
        else:
            prompt = my_input

        model_input = self.encode_input(prompt)

        def generate(model_input: Array, num_tokens: int) -> Array:
            def sample_token(padded_array: Array):
                pad_mask = jnp.where(padded_array == self.pad_token, 0, 1)

                last_tok_idx = jnp.sum(pad_mask) - 1

                if self.args.baseline:
                    logits = inference_model(padded_array, pad_mask, False, self.key)
                else:
                    logits = inference_model(
                        padded_array,
                        self.args.max_iters,
                        pad_mask,
                        False,
                        False,
                        self.key,
                    )[0][-1]

                logits = logits[last_tok_idx, :]  # extract the logits for the last token

                # Apply temperature
                scaled_logits = logits / temperature

                # Apply repetition penalty
                if model_input.shape[0] > max_repeat_tokens:
                    recent_tokens = model_input[-max_repeat_tokens:]
                    scaled_logits = jax.vmap(lambda t: jnp.where(t == jnp.arange(logits.shape[0]),
                                                                 scaled_logits / repetition_penalty,
                                                                 scaled_logits))(recent_tokens)
                    scaled_logits = scaled_logits.min(axis=0)

                # Top-k sampling
                top_k_logits, top_k_indices = jax.lax.top_k(scaled_logits, top_k)

                # Top-p (nucleus) sampling
                probs = jax.nn.softmax(top_k_logits)
                cumulative_probs = jnp.cumsum(probs)
                nucleus = jnp.where(cumulative_probs > top_p, 0.0, probs)
                nucleus = nucleus / jnp.sum(nucleus)

                # Sample from the nucleus
                token = jax.random.choice(self.key, top_k_indices, p=nucleus)

                return token

            for _ in range(num_tokens):
                if model_input.shape[0] < self.args.seqlen:
                    padded_array = jnp.pad(
                        model_input,
                        (0, self.args.seqlen - model_input.shape[0]),
                        constant_values=self.pad_token,
                    )
                else:
                    padded_array = model_input[-self.args.seqlen:]

                gen = sample_token(padded_array)

                model_input = jnp.concatenate([model_input, gen.reshape(-1)])  # append the generated token for AR

                # Update PRNG key
                self.key, _ = jax.random.split(self.key)

            return model_input

        generated_output = generate(model_input, num_tokens)

        return self.decode_fn(generated_output[-num_tokens:])

    def inference(self, my_input: str, num_tokens: int = 32):
        model = self.skeleton_model()

        assert (
            model.__name__ == "GPT" if args.baseline else model.__name__ == "ReAct"
        ), "Trying to do inference on baseline model requires --baseline flag"

        model = load_eqx_obj(self.args.checkpoint_path, model)

        count_params(model)

        output = self.sample_model(
            model=model,
            my_input=my_input,
            num_tokens=num_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_repeat_tokens=3,
        )

        return output


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    args = get_inference_args()
    logger = UnifiedLogger(level="DEBUG")
    my_logger = logger.my_logger()

    my_logger.warning('Make sure to provide the correct args per the model configuration - as it cant be autodetected!')
    my_logger.warning("These are: max_iters | baseline | naive | num_blocks | width | n_heads")
    print(f"{'-'*50}\n")

    assert args.checkpoint_path is not None, "Please provide a checkpoint path"
    assert os.path.exists(args.checkpoint_path), "Please provide a valid checkpoint path | File does not exist"
    assert args.prompt is not None, "Please provide a prompt/input for inference"

    inferencer = Inferencer(args, key)

    output = inferencer.inference(args.prompt, args.num_tokens)

    print(f"\n{'~' * 50}\n")
    my_logger.info(f'System Prompt: {args.system_prompt}\n')
    my_logger.info(f'User Prompt: {args.prompt}\n')
    my_logger.info(f'Model Response: {output}\n')
    print(f"\n{'~' * 50}\n")
