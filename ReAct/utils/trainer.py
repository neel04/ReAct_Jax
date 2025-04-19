import os
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import optuna
from jaxtyping import Array, Int, PRNGKeyArray, PyTree
from jmp import Policy
from optax._src.base import GradientTransformation
from tqdm.auto import tqdm

import wandb
from inferencer import Inferencer
from ReAct.model.baseline import GPT
from ReAct.model.blocks import LinearProj
from ReAct.model.react import React
from ReAct.utils.helpers import (
    Profiler,
    calc_performance_metrics,
    count_params,
    get_hist,
    get_weights,
    load_eqx_obj,
    megatron_init,
    save_eqx_obj,
)
from ReAct.utils.losses import (
    _cross_entropy_with_logits_bwd,
    _cross_entropy_with_logits_fwd,
    cross_entropy_with_logits,
)
from ReAct.utils.muon_modded import muon
from ReAct.utils.sharding import get_strategy

get_linear_weights = partial(get_weights, layer=LinearProj)
half, full = jnp.bfloat16, jnp.float32
policy = Policy(compute_dtype=half, param_dtype=half, output_dtype=half)

# Stable CE (w/ z-loss) from PaLM
ce_loss = cross_entropy_with_logits
ce_loss.defvjp(_cross_entropy_with_logits_fwd, _cross_entropy_with_logits_bwd)

@eqx.filter_jit
def iters_fwd(
    model: React, input_arr: Array, pad_mask: Array, iters_to_do: int, key: PRNGKeyArray
) -> Array:
    # Only n passes, but track the gradient
    output, _ = model(
        input_arr,
        iters_to_do=iters_to_do,
        pad_mask=pad_mask,
        prev_thought=False,
        is_training=True,
        key=key,
    )

    return output

@eqx.filter_jit
def vanilla_fwd(
    model: GPT, input_arr: Array, pad_mask: Array, iters_to_do: int, key: PRNGKeyArray
) -> Array:
    return model(input_arr, pad_mask, enable_dropout=True, key=key)

@eqx.filter_jit
def _compute_softmax_cross_entropy_loss(pred_y: Array, y_one_hot: Array) -> Array:
    loss, _  = ce_loss(pred_y, y_one_hot) # (batch_size, seqlen)

    return loss.mean()

@eqx.filter_jit
def make_step(
    keys: PRNGKeyArray,
    model: Union[React, GPT],
    opt_state: PyTree,
    filter_spec: PyTree,
    x: Array,
    y: Array,
    pad_mask: Array,
    iters_to_do: int,
    optim: GradientTransformation,
    num_classes: int,
):
    x, y, pad_mask = strategy.shard_cast((x, y, pad_mask))
    model, opt_state = strategy.shard_model((model, opt_state))
    dynamic_model = eqx.filter(model, eqx.is_inexact_array)

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def compute_loss(
        model: React | GPT,
        x: Array,
        y: Array,
        pad_mask: Array,
        iters_to_do: int,
        num_classes: int,
        keys: PRNGKeyArray,
    ) -> Array:
        """
        Computes the loss of the model w.r.t the input.
        """
        if model.__name__ == "ReAct":
            forward = iters_fwd
        else:
            forward = vanilla_fwd

        pred_y = jax.vmap(forward, in_axes=(None, 0, 0, None, 0))(
            model, x, pad_mask, iters_to_do, keys
        )  # (batch_size, seqlen, num_classes)

        y_one_hot = jax.nn.one_hot(
            y, num_classes=num_classes
        )  # (batch_size, seqlen, num_classes)

        loss = _compute_softmax_cross_entropy_loss(pred_y, y_one_hot)

        return loss

    loss, grads = compute_loss(model, x, y, pad_mask, iters_to_do, num_classes, keys)
    grads = strategy.shard_model_cast(grads)  # cast to bfloat16
    updates, opt_state = optim.update(grads, opt_state, dynamic_model)
    updates = strategy.shard_model(updates)
    model = eqx.apply_updates(model, updates)

    # shard the updated state as well
    model, opt_state = strategy.shard_model((model, opt_state))

    return loss, model, opt_state, grads, updates


class Trainer:
    def __init__(
        self,
        args: Any,
        loggers: Tuple,
        loaders: Tuple,
        decode_fn: Callable,
        dataset_size: Optional[int] = None,
        key: PRNGKeyArray = jax.random.PRNGKey(69),
    ):

        global strategy
        strategy = get_strategy(args.strategy, args.model_axis)
        strategy._policy = policy # register the policy

        self.decode_fn = decode_fn # decode the ids to text
        self.text_data: list = []
        self.args = args
        self.key = key

        self.my_logger, self.wandb_logger = loggers
        self.trainloader, self.valloader = loaders

        self.dataset_length = (
            dataset_size if dataset_size is not None else len(self.trainloader)
        )

        self.my_logger.info(f"Using Args: {self.args}\n")

    def evaluate_acc(
        self,
        model: Union[React, GPT],
        is_baseline: bool,
        loader: Any,
        eval_iters: int,
        keys: PRNGKeyArray,
    ):

        model = strategy.shard_model((model))

        metrics_sum = jnp.zeros(3)  # [acc, loss, ppl]
        num_batches = len(loader)

        for _, batch in tqdm(enumerate(loader), total=len(loader), desc='Validating'):
            seq, label, pad_mask = jnp.asarray(batch['text'])
            seq, label, pad_mask = policy.cast_to_compute((seq, label, pad_mask))
            seq, label, pad_mask = strategy.shard_cast((seq, label, pad_mask))

            acc, loss, ppl = self.compute_metrics(
                keys,
                model,
                is_baseline,
                seq,
                label,
                pad_mask,
                eval_iters,
                self.args.num_classes,
            )

            metrics_sum += jnp.array([acc, loss, ppl])

        # Compute cumulatives
        cum_acc, cum_loss, cum_ppl = metrics_sum / num_batches

        return (cum_acc, cum_loss, cum_ppl), seq[0]  # type: ignore

    def set_optim_and_scheduler(
        self, model: eqx.Module
    ) -> Tuple[GradientTransformation, PyTree, eqx.Module]:

        assert model is not None, "Model is not initialized"

        total_steps = self.args.epochs * self.dataset_length

        self.schedule_fn = optax.warmup_cosine_decay_schedule(
            init_value=self.args.lr / 2,
            peak_value=self.args.lr,
            end_value=self.args.lr / 10,
            warmup_steps=self.args.warmup_steps,
            decay_steps=total_steps,
        )

        # optimizer with weight decay
        match self.args.optimizer_type:
            case "muon":
                opt = muon(
                    learning_rate=self.schedule_fn,
                    adam_weight_decay=self.args.weight_decay,
                    adam_b1=self.args.beta_1,
                    adam_b2=self.args.beta_2,
                    nesterov=self.args.nesterov,
                    # adaptive=self.args.muon_is_adaptive,
                    adaptive=True,
                )

            case _:
                self.my_logger.warning("Using AdamW optimizer.")

                opt = optax.adamw(
                    learning_rate=self.schedule_fn,
                    weight_decay=self.args.weight_decay,
                    b1=self.args.beta_1,
                    b2=self.args.beta_2,
                    nesterov=self.args.nesterov,
                )

        optim = optax.chain(
            optax.adaptive_grad_clip(self.args.grad_clip),
            optax.MultiSteps(
                opt,
                every_k_schedule=self.args.accum_steps,
            ),
        )

        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        return optim, opt_state, model

    @staticmethod
    def get_filterspec(model: eqx.Module) -> PyTree[bool]:
        """
        Returns a filter spec for the model to filter out the trainable parameters.
        Can be used to freeze or unfreeze certain modules of the model depending on the step and epoch.

        Args:
            model: The model to filter
        Returns:
            filter_spec: The filter spec as a PyTree[bool] marking trainable parameters
        """
        filter_spec = jax.tree_util.tree_map(lambda _: True, model)  # all trainable

        return filter_spec

    def init_model(self, key: PRNGKeyArray) -> Tuple[PyTree, Union[React, GPT]]:

        if self.args.baseline:
            self.max_iters = 1 # baseline model only does one pass

            model = GPT(
                self.args.n_heads,
                self.args.seqlen,
                self.args.num_blocks,
                self.args.width,
                self.args.drop_rate,
                self.args.num_classes,
                key,
                strategy
            )
        else:
            model = React(
                self.args.n_heads,
                self.args.seqlen,
                self.args.max_iters,
                self.args.num_blocks,
                self.args.width,
                self.args.drop_rate,
                self.args.num_classes,
                key,
                strategy
            )

        # custom weight init
        weights= get_linear_weights(model)

        new_weights = [
            megatron_init(weight, subkey)
            for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
        ]

        model = eqx.tree_at(get_linear_weights, model, new_weights)

        # switch to half precision
        if self.args.bf16:
            model = policy.cast_to_param(model)

        _, opt_state, model = self.set_optim_and_scheduler(model)

        model = strategy.shard_model(model)
        count_params(model)  # prints to stdout
        calc_performance_metrics(self.args, self.my_logger)  # logs via logger

        return opt_state, model

    def resume_training(
        self, model: PyTree, opt_state: eqx.Module
    ) -> tuple[PyTree, PyTree, int, int]:
        if isinstance(self.args.resume, str):
            run_path, epoch, step = self.args.resume.split("+")
            run_path, epoch, step = run_path.strip(), int(epoch.strip()), int(step.strip())

            base_path = "https://api.wandb.ai/files/"
            model_path = f'{base_path}{run_path}/model_{epoch}_{step}.eqx'

            # wget both files to ReAct/outputs/, if those files don't exist
            if not os.path.exists(f'{self.args.save_dir}model_{epoch}_{step}.eqx'):
                os.system(f'wget -O {self.args.save_dir}model_{epoch}_{step}.eqx {model_path}')
        else:
            # get the model with max step & epoch number living in `save_dir`
            files = [
                os.path.join(self.args.save_dir, file)
                for file in os.listdir(self.args.save_dir)
                if file.endswith("eqx")
            ]

            get_info = lambda idx: os.path.basename(latest_file).split(".")[0].split("_")[idx]  # noqa: E731
            latest_file = max(files, key=os.path.getctime)
            step, epoch = int(get_info(-1)), int(get_info(-2))

        model, opt_state = load_eqx_obj( f"{self.args.save_dir}model_{epoch}_{step}.eqx", (model, opt_state) )

        self.my_logger.info(f'\n-------- Resuming training from step {step} ---------\n')

        return model, opt_state, step, epoch

    @eqx.filter_jit
    def compute_metrics(
        self,
        keys: PRNGKeyArray,
        model: eqx.Module,
        is_baseline: bool,
        input_arr: Array,
        label: Array,
        pad_mask: Array,
        eval_iters: int,
        num_classes: int,
    ) -> tuple[Array, Int[Array, "1"], Array]:
        """
        Computes the accuracy, perplexity, loss of the model w.r.t batch
        """
        # sharding everything
        model = strategy.shard_model(model)
        input_arr, label, pad_mask = strategy.shard_cast((input_arr, label, pad_mask))

        keys = keys[:input_arr.shape[0], ...] # take a batch_size sized slice of the keys

        if is_baseline:
            pred_y = jax.vmap(model, in_axes=(0, 0, None, 0))(input_arr, pad_mask, False, keys) # type: ignore
        else:
            pred_y = jax.vmap(model, in_axes=(0, None, 0, None, None, 0))(input_arr, eval_iters, pad_mask, False, False, keys)[0] # type:ignore

        y_hat = jax.nn.softmax(pred_y, axis=-1).argmax(-1)

        # compute accuracy
        accuracy = jnp.mean(y_hat == label)

        # compute loss
        y_one_hot = jax.nn.one_hot(label, num_classes=num_classes) # (batch_size, seqlen, num_classes)
        loss = ce_loss(pred_y, y_one_hot)[0].mean()

        # compute perplexity
        perplexity = jnp.exp(loss)

        return accuracy, loss, perplexity

    def optuna_log(self, trial: Optional[Any], metrics: Tuple[float, int]):
        '''
        Logs the metrics to the optuna trial
        '''
        loss, progress = metrics

        if trial is not None:
            trial.report(loss, step=progress)
            self.my_logger.info(f"\nReported metric: {loss} @ {progress} to optuna.")

            if trial.should_prune():
                self.wandb_logger.finish() # finish before pruning.
                raise optuna.exceptions.TrialPruned()

    def train(self, trial: Optional[Any] = None) -> float:
        step_done, epoch_done, val_loss = 0, 0, 999.9

        prof = Profiler(self.args.profile)
        opt_state, model = self.init_model(self.key)
        optim, _, _ = self.set_optim_and_scheduler(model)
        filter_spec = self.get_filterspec(model)

        if self.args.resume is True and self.args.tune_hyperparams is False:
            model, opt_state, step_done, epoch_done = self.resume_training(
                model, opt_state
            )

        print(f"Model: {model}")

        for epoch in range(epoch_done, self.args.epochs):
            train_acc, train_loss, train_ppl = [], [], []

            epoch_key = jnp.array([epoch, epoch + 1]).astype(jnp.uint32)
            keys = jax.random.split(epoch_key, self.args.batch_size)

            for step, batch in tqdm(enumerate(self.trainloader), total=self.dataset_length, desc=f'Epoch {epoch}'):
                step += step_done  # for multiple epochs
                prof.start_prof(step)

                seq, label, pad_mask = jnp.asarray(batch["text"])
                seq, label, pad_mask = policy.cast_to_compute((seq, label, pad_mask))
                seq, label, pad_mask = strategy.shard_cast((seq, label, pad_mask))

                loss, model, opt_state, grads, updates = make_step(
                    keys=keys,
                    model=model,
                    opt_state=opt_state,
                    filter_spec=filter_spec,
                    x=seq,
                    y=label,
                    pad_mask=pad_mask,
                    iters_to_do=self.args.max_iters,
                    optim=optim,
                    num_classes=self.args.num_classes,
                )

                loss = prof.stop_prof(loss, step)  # end trace if profiled

                if step % 100 == 0:
                    accuracy, loss, perplexity = self.compute_metrics(
                        keys=keys,
                        model=model,
                        is_baseline=self.args.baseline,
                        input_arr=seq,
                        label=label,
                        pad_mask=pad_mask,
                        eval_iters=self.args.max_iters,
                        num_classes=self.args.num_classes,
                    )

                    train_acc.append(accuracy)
                    train_loss.append(loss)
                    train_ppl.append(perplexity.mean())

                    loss = loss.item()

                    self.wandb_logger.log(
                        {
                            "Train/loss": loss,
                            "Train/Lr": self.schedule_fn(epoch + 1 * step).item(),  # type: ignore
                            "Train/tokens": step * self.args.batch_size * self.args.seqlen,
                        },
                        step=step,
                    )

                    if jnp.isnan(loss):
                        self.my_logger.warning(f"\nLoss is NaN at step {step}")
                        self.wandb_logger.finish()
                        return loss

                if step % self.args.log_interval == 0 and len(train_acc) > 0:
                    # Compute cumulatives
                    cum_train_acc = sum(train_acc) / len(train_acc)
                    cum_train_loss = sum(train_loss) / len(train_loss)
                    cum_train_ppl = sum(train_ppl) / len(train_ppl)

                    # clear the metrics
                    train_acc, train_loss, train_ppl = [], [], []

                    ## Validation
                    (val_acc, val_loss, val_ppl), val_sample = self.evaluate_acc(
                        model,
                        self.args.baseline,
                        self.valloader,
                        self.args.max_iters,
                        keys,
                    )

                    self.wandb_logger.log(
                        {
                            "Train/acc": cum_train_acc,
                            "Train/cum_loss": cum_train_loss,
                            "Train/ppl": cum_train_ppl,
                            "Val/acc": val_acc,
                            "Val/loss": val_loss,
                            "Val/ppl": val_ppl,
                            "Gradients": wandb.Histogram(np_histogram=get_hist(grads)),
                            "Updates": wandb.Histogram(np_histogram=get_hist(updates)),
                            "Weights": wandb.Histogram(np_histogram=get_hist(model)),
                        },
                        step=step,
                    )

                    # Report metrics to optuna
                    self.optuna_log(trial, (val_loss, step))

                    ## Visualize one sample and model prediction
                    sample_x, val_sample_x = seq[0][:16], val_sample[:16]

                    self.my_logger.info(f"epoch={epoch}, step={step}, loss={loss}")
                    self.my_logger.info(
                        f"Validation accuracy: {val_acc} | using {self.args.max_iters} iterations"
                    )
                    self.my_logger.info(
                        f"Cumulative Training accuracy: {cum_train_acc}\n"
                    )

                    self.generate_from_model(
                        model,
                        sample_x,
                        metadata={"type": "train", "step": step},
                        max_new_tokens=64,
                    )
                    self.generate_from_model(
                        model,
                        val_sample_x,
                        metadata={"type": "val", "step": step},
                        max_new_tokens=64,
                    )

                    jax.experimental.multihost_utils.sync_global_devices(  # type: ignore
                        "Sync up all nodes after inference."
                    )

                if not self.args.tune_hyperparams and (step + 1) % self.args.save_interval == 0:
                    filepath = f"{self.args.save_dir}model_{epoch}_{step}.eqx"

                    save_eqx_obj(self.args.save_dir, filepath, (model, opt_state))

                    self.my_logger.info(f"Model saved at {filepath}")
                    self.wandb_logger.save(filepath)

            step_done = step  # type: ignore
            self.optuna_log(trial, (val_loss, step))  # type: ignore

            print(f"Epoch {epoch} done!")

        self.wandb_logger.finish()

        return val_loss

    def generate_from_model(
        self,
        model: eqx.Module,
        input_arr: Array,
        metadata: dict,
        max_new_tokens: int,
        temperature: float = 0.5,
    ):
        decoded_input: str = self.decode_fn(input_arr)
        prompt = f"Prompt: {decoded_input}"

        inferencer = Inferencer(self.args, self.key)
        model_gen = inferencer.sample_model(
            model, decoded_input, max_new_tokens, temperature
        ).strip()

        # Format the generated text
        model_gen = f"model generation: {model_gen}\n"

        self.my_logger.info(prompt)
        self.my_logger.info(model_gen)

        # Log prompts-gen pairs to wandb
        new_table = wandb.Table(columns=["Step", "Prompt", "Model Generation", "Type"])
        new_table.add_data(metadata["step"], prompt, model_gen, metadata["type"])
        self.wandb_logger.log({"Generated Samples": new_table})
