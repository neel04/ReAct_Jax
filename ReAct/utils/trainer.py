import os
from functools import partial
from typing import Callable, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree
from scalax.sharding import MeshShardingHelper
from scalax.sharding import PartitionSpec as P
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from ReAct.model.baseline import GPT
from ReAct.model.react import React
from ReAct.utils.helpers import count_params, load_eqx_obj, save_eqx_obj

from .helpers import broad_to_bsz, calc_performance_metrics, half_precision, xla_calc_flops

mesh = MeshShardingHelper(axis_dims=[-1], axis_names=['data']) # handle DDP + TP over multi-node

@eqx.filter_jit
def n_k_loop(model: eqx.Module, input_arr: Array, pad_mask: Array, n: Array, k: Array, key: PRNGKeyArray) -> Array:
    key1, key2 = jax.random.split(key, 2)

    # forward pass the model without tracking grads
    _, intermediate_array = model(
        input_arr,
        iters_to_do=n,
        pad_mask=pad_mask,
        prev_thought=False,
        is_training=True,
        key=key1)

    intermediate_array = jax.lax.stop_gradient(intermediate_array)

    # k passes but track the gradient
    output, _ = model(
        (input_arr, intermediate_array),
        iters_to_do=k,
        pad_mask=pad_mask,
        prev_thought=True,
        is_training=True,
        key=key2)

    return output

@eqx.filter_jit
def iters_fwd(model: eqx.Module, input_arr: Array, pad_mask: Array, n: int, k: int, key: PRNGKeyArray) -> Array:
    # Only n passes, but track the gradient
    output, _ = model(input_arr,
                      iters_to_do=5,
                      pad_mask=pad_mask,
                      prev_thought=False,
                      is_training=True,
                      key=key)

    return output

@eqx.filter_jit
def vanilla_fwd(model: eqx.Module, input_arr: Array, pad_mask: Array, n: int, k: int, key: PRNGKeyArray) -> Array:
    return model(input_arr, pad_mask, enable_dropout=True, key=key)

@eqx.filter_jit
def _compute_softmax_cross_entropy_loss(pred_y: Array, y_one_hot: Array,
                                        pad_mask: Array, n: Array, k: Array) -> Array:

    loss = -jnp.sum(jax.nn.log_softmax(pred_y, axis=-1) * y_one_hot, axis=-1)

    k = jnp.repeat(k[:, None], loss.shape[1], axis=-1)

    loss = (loss * k).sum(-1) # across the sequence

    return loss.mean() # across all the batches

@partial(
    mesh.sjit,
    in_shardings=(None, None, P('data'), P('data'), P('data')) + (None,) * 3,
    args_sharding_constraint=(None, None, P('data'), P('data'), P('data')) + (None,) * 3,
    out_shardings=None,
    static_argnums=(2, 8, 9)
)
def make_step(model: eqx.Module,
              opt_state: Tuple[PyTree],
              filter_spec: PyTree, # static
              x: Array,
              y: Array,
              pad_mask: Array,
              n: Array,
              k: Array,
              optim: Callable, # static
              num_classes: int, # static
              keys: List[PRNGKeyArray]):

    @eqx.filter_value_and_grad
    def compute_loss(model: eqx.Module, static_model: PyTree, x: Array, y: Array, pad_mask: Array,
                    n: int, k: int, num_classes: int, keys: PRNGKeyArray) -> int:
        '''
        Computes the loss of the model w.r.t the input. Is a closure for accessing static_model
        '''
        model = eqx.combine(model, static_model)

        if model.__name__ == 'ReAct':
            forward = iters_fwd
        else:
            forward = vanilla_fwd

        pred_y = jax.vmap(forward, in_axes=(None, 0, 0, 0, 0, 0))(model, x, pad_mask, n, k, keys) # (batch_size, seqlen, num_classes)
        y_one_hot = jax.nn.one_hot(y, num_classes=num_classes) # (batch_size, seqlen, num_classes)
        loss = _compute_softmax_cross_entropy_loss(pred_y, y_one_hot, pad_mask, n, k)

        return loss

    diff_model, static_model = eqx.partition(model, filter_spec,
                                             is_leaf=lambda x: isinstance(x, eqx.nn.Dropout))

    loss, grads = compute_loss(diff_model, static_model, x, y, pad_mask, n, k, num_classes, keys)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return loss, model, opt_state

class Trainer:
    def __init__(self,
                 args: dict,
                 logger: Tuple,
                 loaders: Tuple,
                 decode_fn: Callable,
                 key: PRNGKeyArray = jax.random.PRNGKey(69)):

        self.decode_fn = decode_fn # decode the ids to text
        self.args = args
        self.key = key

        # unpacking the loaders & loggers
        self.my_logger, self.wandb_logger = logger
        self.trainloader, self.valloader = loaders
        self.dataset_length = len(self.trainloader) * args.batch_size * args.seqlen

        self.my_logger.info(f'Using Args: {self.args}\n')

        # Assign each arg as a class attribute
        for k, v in vars(self.args).items():
                setattr(self, k, v)

    def get_n_k(self, key: PRNGKeyArray) -> Tuple[Array, Array]:
        n_key, k_key = jax.random.split(key, 2)

        # Getting the random n and k
        rndm_n = jax.random.randint(n_key, shape=(1,), minval=1, maxval=self.max_iters)
        rndm_k = jax.random.randint(k_key, shape=(1,), minval=rndm_n.item(), maxval=self.max_iters - rndm_n.item() + 1)

        # Broadcast the (1,) array to batch_size and clip if needed
        rndm_n, rndm_k = broad_to_bsz(rndm_n, (self.batch_size,)), broad_to_bsz(rndm_k, (self.batch_size,))
        rndm_n, rndm_k = jnp.clip(rndm_n, 1, self.max_iters), jnp.clip(rndm_k, 1, self.max_iters)

        return rndm_n.astype(int), rndm_k.astype(int)

    def evaluate_acc(self, model: eqx.Module, loader: DataLoader, eval_iters: int, keys: List[PRNGKeyArray]):

        metric = []

        for step, batch in tqdm(enumerate(loader), total=len(loader), desc='Validating'):
            seq, label, pad_mask = jnp.asarray(batch['text'])
            acc, loss, ppl = self.compute_metrics(model, seq, label, pad_mask, eval_iters, self.num_classes, keys)
            metric.extend([acc, loss, ppl])

        # Compute cumulatives
        cum_acc = sum(metric[::3]) / len(metric[::3])
        cum_loss = sum(metric[1::3]) / len(metric[1::3])
        cum_ppl = sum(metric[2::3]) / len(metric[2::3])

        return (cum_acc, cum_loss, cum_ppl), seq[0] # return one sample for viz

    def set_optim_and_scheduler(self, model: eqx.Module) -> Tuple[Callable, PyTree, eqx.Module]:
        assert model is not None, 'Model is not initialized'

        total_steps = self.epochs * len(self.trainloader)

        self.schedule_fn = optax.warmup_cosine_decay_schedule(init_value=self.lr / 2,
                                                              peak_value=self.lr,
                                                              end_value=self.lr / 10,
                                                              warmup_steps=self.warmup_steps,
                                                              decay_steps=total_steps)

        # optimizer with weight decay
        optim = optax.chain(
            optax.adamw(learning_rate=self.schedule_fn, weight_decay=self.weight_decay),
            optax.clip_by_global_norm(self.grad_clip),
            optax.apply_every(self.accum_steps)
        )

        opt_state = optim.init(eqx.filter(model, eqx.is_array_like))

        return optim, opt_state, model

    @staticmethod
    def get_filterspec(model: eqx.Module) -> PyTree[bool]:
        '''
        Returns a filter spec for the model to filter out the trainable parameters.
        Can be used to freeze or unfreeze certain modules of the model depending on the step and epoch.

        Args:
            model: The model to filter
        Returns:
            filter_spec: The filter spec as a PyTree[bool] marking trainable parameters
        '''
        filter_spec = jax.tree_util.tree_map(lambda _: True, model) # all trainable
        filter_spec = eqx.tree_at(
            lambda tree: tree.pos_enc, # pos_enc should be frozen
            filter_spec,
            replace=False)

        return filter_spec

    def init_model(self, key: PRNGKeyArray):

        if self.baseline:
            model = GPT(self.n_heads, self.seqlen, self.num_blocks, self.width,
                        self.drop_rate, self.num_classes, key)
        else:
            model = React(self.n_heads, self.seqlen, self.max_iters, self.num_blocks, self.width,
                           self.drop_rate, self.num_classes, key)

        # switch to half precision
        if self.bf16:
            model = half_precision(model)

        _, opt_state, model = self.set_optim_and_scheduler(model)
        
        count_params(model) # prints to stdout
        calc_performance_metrics(self.args, self.my_logger) # logs via logger

        return opt_state, model

    def resume_training(self, model: eqx.Module, opt_state: eqx.Module):
        # extracting out the paths
        run_path, step = self.resume.split('+')
        run_path, step = run_path.strip(), int(step.strip())

        base_path = "https://api.wandb.ai/files/"
        model_path = f'{base_path}{run_path}/model_{step}.eqx'

        # wget both files to ReAct/outputs/, if they those files don't exist
        if not os.path.exists(f'{self.save_dir}model_{step}.eqx'):
            os.system(f'wget -O {self.save_dir}model_{step}.eqx {model_path}')

        model, opt_state = load_eqx_obj(f'{self.save_dir}model_{step}.eqx', (model, opt_state))

        self.my_logger.info(f'-------- Resuming training from step {step} ---------\n')

        return model, opt_state, step

    @partial(
        mesh.sjit,
        in_shardings=(None, P('data'), P('data'), P('data'), None),
        out_shardings=None,
        args_sharding_constraint=(None, P('data'), P('data'), P('data'), None),
        static_argnums=(0, 5, 6)
    )
    def compute_metrics(self,
                        model: eqx.Module,
                        input_arr: Array,
                        label: Array,
                        pad_mask: Array,
                        eval_iters: int, # static
                        num_classes: int, # static
                        keys: List[PRNGKeyArray]):
        '''
        Computes the accuracy, perplexity, loss of the model w.r.t batch
        '''
        eval_iters = jnp.ones_like(input_arr[:, 0]) * eval_iters
        keys = keys[:input_arr.shape[0], ...] # take a batch_size sized slice of the keys

        if self.baseline:
            pred_y = jax.vmap(model, in_axes=(0, 0, None, 0))(input_arr, pad_mask, False, keys)
        else:
            pred_y = jax.vmap(model, in_axes=(0, 0, 0, None, None, 0))(input_arr, eval_iters, pad_mask, False, False, keys)[0]

        # compute accuracy
        y_hat = jax.nn.softmax(pred_y, axis=-1).argmax(-1) * pad_mask
        accuracy = jnp.mean(y_hat == label)

        # compute loss
        y_one_hot = jax.nn.one_hot(label, num_classes=num_classes) # (batch_size, seqlen, num_classes)
        loss = optax.softmax_cross_entropy(pred_y, y_one_hot).mean()

        # compute perplexity
        perplexity = jnp.exp(loss)

        return accuracy, loss, perplexity

    def train(self):
        step_done = 0
        
        rndm_n, rndm_k = self.get_n_k(key=self.key) # initial n and k
        dummy_arr = jax.ShapeDtypeStruct((self.batch_size, self.seqlen), jnp.uint16)
        
        opt_state, model = self.init_model(self.key)
        optim, _, _ = self.set_optim_and_scheduler(model)
        filter_spec = self.get_filterspec(model)

        if self.resume:
            model, opt_state, epoch_done = self.resume_training(model, opt_state)
        else:
            epoch_done = 0

        print(f'Model: {model}')
        
        # Report XLA computed FLOPs to do the make_step 
        xla_calc_flops(
            fn=make_step,
            args=(model, opt_state, filter_spec,
                  dummy_arr, dummy_arr, dummy_arr,
                  rndm_n, rndm_k, optim, self.num_classes,
                  jax.random.split(self.key, self.batch_size)),
            static_argnums=(2, 8, 9),
            my_logger=self.my_logger
        )

        for epoch in range(epoch_done, self.epochs):
            # init empty metrics
            epoch_key = jnp.array([epoch, epoch + 1]).astype(jnp.uint32)
            train_acc, train_loss, train_ppl = [], [], []

            keys = jax.random.split(epoch_key, self.batch_size)
            
            for step, batch in tqdm(enumerate(self.trainloader), total=len(self.trainloader), desc=f'Epoch {epoch}'):
                step += step_done # for multiple epochs

                seq, label, pad_mask = jnp.asarray(batch['text'])

                loss, model, opt_state = make_step(model, opt_state, filter_spec, seq, label, pad_mask,
                                                   rndm_n, rndm_k, optim, self.num_classes, keys)

                if step % 100 == 0:
                    #rndm_n, rndm_k = self.get_n_k(key=keys[step % self.batch_size])
                    
                    accuracy, loss, perplexity = self.compute_metrics(model, seq, label, pad_mask,
                                                                      self.max_iters, self.num_classes, keys)

                    train_acc.append(accuracy)
                    train_loss.append(loss)
                    train_ppl.append(perplexity.mean())

                    loss = loss.item()

                    self.wandb_logger.log(
                        {
                            'Train/loss': loss,
                            'Train/Lr': self.schedule_fn(epoch + 1 * step).item()
                        },
                        step=step
                    )

                    # Terminate if loss is NaN
                    if jnp.isnan(loss):
                        self.my_logger.warning(f'\nLoss is NaN at step {step}')
                        return loss

                if (step + 1) % self.log_interval == 0 and len(train_acc) > 0:
                    # Compute cumulatives
                    cum_train_acc = sum(train_acc) / len(train_acc)
                    cum_train_loss = sum(train_loss) / len(train_loss)
                    cum_train_ppl = sum(train_ppl) / len(train_ppl)

                    ## clear the metrics
                    train_acc, train_loss, train_ppl = [], [], []

                    ## Validation
                    val_metrics, val_sample = self.evaluate_acc(model, self.valloader, self.max_iters, keys)

                    self.wandb_logger.log(
                        {
                            'Train/acc': cum_train_acc,
                            'Train/cum_loss': cum_train_loss,
                            'Train/ppl': cum_train_ppl,
                            'Val/acc': val_metrics[0],
                            'Val/loss': val_metrics[1],
                            'Val/ppl': val_metrics[2],
                        },
                        step=step
                    )

                    ## Visualize one sample and model prediction
                    sample_x, val_sample_x = seq[0][:16], val_sample[:16]

                    self.my_logger.info(f"epoch={epoch}, step={step}, loss={loss}")
                    self.my_logger.info(f'Validation accuracy: {val_metrics[0]} | using {self.max_iters} iterations')
                    self.my_logger.info(f'Cumulative Training accuracy: {cum_train_acc}\n')

                    self.generate(model, sample_x, metadata={'type': 'train', 'step': step}, max_new_tokens=96)
                    self.generate(model, val_sample_x, metadata={'type': 'val', 'step': step}, max_new_tokens=96)

                if not self.tune_hyperparams and step % self.save_interval == 0:
                    filepath = f"{self.save_dir}model_{epoch}_{step}.eqx"
                    
                    #model = jax.experimental.multihost_utils.process_allgather(model) # all-gather the model
                    save_eqx_obj(self.save_dir, filepath, (model, opt_state))
                    
                    self.my_logger.info(f'Model saved at {filepath}')
                    self.wandb_logger.save(filepath)

            print(f'Epoch {epoch} done!')
            step_done = step # prepare for next epoch

        self.wandb_logger.finish()
        return loss

    def generate(self, model: eqx.Module, input_arr: Array, metadata: dict, max_new_tokens: int, temperature: float = 0.5):
        '''
        Take a conditioning sequence, call output_head to obtain a prediction
        and autoregressively complete the sequence max_new_tokens times.
        '''
        key = jax.random.PRNGKey(0)
        inference_model = eqx.nn.inference_mode(model)

        text_table = wandb.Table(columns=["Step", "Prompt", "Model Generation", "Type"])
        prompt = f'Prompt: {self.decode_fn(input_arr)}'

        for _ in range(max_new_tokens):
            if input_arr.shape[0] < self.seqlen:
                padded_array = jnp.pad(input_arr, (0, self.seqlen - input_arr.shape[0]))
            else:
                padded_array = input_arr

            pad_mask = jnp.array([1 if i != 0 else 0 for i in padded_array])

            # Find the correct index to extract the logits from
            try:
                zero_idx = jnp.where(padded_array == 0)[0][0]
            except IndexError:
                zero_idx = self.seqlen

            if self.baseline:
                logits = inference_model(padded_array, pad_mask, False, key)
            else:
                logits = inference_model(padded_array, self.max_iters, pad_mask, False, False, key)[0]
            
            logits = logits[zero_idx - 1, :] # extract the logits for the last token
            gen = jax.nn.softmax(logits / temperature).argmax() # greedy decoding
            input_arr = jnp.concatenate([input_arr, gen.reshape(-1)]) # append the generated token for AR

        model_gen = f'model generation: {self.decode_fn(input_arr[-max_new_tokens:-1])}\n'
        self.my_logger.info(prompt)
        self.my_logger.info(model_gen)

        # log to logger as a table
        text_table.add_data(metadata['step'], prompt, model_gen, metadata['type'])
        self.wandb_logger.log({'Generated Samples': text_table})

        return input_arr
