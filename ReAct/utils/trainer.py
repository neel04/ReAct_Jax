import os
from typing import Any, Callable, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax.experimental.compilation_cache import compilation_cache
from jaxtyping import Array, PRNGKeyArray
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ReAct.model.react import React
from ReAct.model.baseline import GPT
from ReAct.utils.helpers import count_params, load_eqx_obj, save_eqx_obj
from ReAct.utils.logger import UnifiedLogger

from .helpers import get_rand_nums, half_precision

compilation_cache.initialize_cache('./compilation_cache')

@eqx.filter_jit
def n_k_loop(model: eqx.Module, input_arr: Array, pad_mask: Array, n: int, k: int, key: PRNGKeyArray) -> Array:
    # Only k passes, but track the gradient
    output_k, _ = model(input_arr, k, pad_mask=pad_mask, key=key)
   
    return output_k

@eqx.filter_jit
def vanilla_fwd(model: eqx.Module, input_arr: Array, pad_mask: Array, n: int, k: int, key: PRNGKeyArray) -> Array:
    return model(input_arr, pad_mask, key=key)

@eqx.filter_value_and_grad
def compute_loss(model: eqx.Module, x: Array, y: Array, pad_mask: Array,
                 n: int, k: int, num_classes: int = 2, keys: PRNGKeyArray = None):
    
    if model.__name__ == 'ReAct':
        forward = n_k_loop
    else:
        forward = vanilla_fwd
    
    pred_y = jax.vmap(forward, in_axes=(None, 0, 0, 0, 0, 0))(model, x, pad_mask, n, k, keys) # (batch_size, seqlen, num_classes)
    
    y_one_hot = jax.nn.one_hot(y, num_classes=num_classes) # (batch_size, seqlen, num_classes)
    
    loss = _compute_softmax_cross_entropy_loss(pred_y, y_one_hot, pad_mask, n, k)
    
    return loss

@eqx.filter_jit
def _compute_softmax_cross_entropy_loss(pred_y: Array, y_one_hot: Array,
                                        pad_mask: Array, n: Array, k: Array) -> Array:
    
    loss = -jnp.sum(jax.nn.log_softmax(pred_y, axis=-1) * y_one_hot, axis=-1)
    
    k = jnp.repeat(k[:, None], loss.shape[1], axis=-1)

    loss = (loss * k).sum(-1) # across the sequence
    
    return loss.mean() # across all the batches

@eqx.filter_jit
def make_step(model: eqx.Module, x: Array, y: Array, pad_mask: Array, n: int, k: int,
              optim, opt_state, num_classes: int, keys: List[PRNGKeyArray]):  # noqa: F821
    
    loss, grads = compute_loss(model, x, y, pad_mask, n, k, num_classes, keys)
    updates, opt_state = optim.update(grads, opt_state, model)
    
    model = eqx.apply_updates(model, updates)
    
    return loss, model, opt_state

class Trainer:
    def __init__(self, args: dict, logger, decode_fn: Callable, shard: Any = None):
        
        self.shard = shard if shard is not None else None
        self.dataset_length = 2119719
        self.decode_fn = decode_fn # decode the ids to text
        
        logger = UnifiedLogger(args, level='DEBUG')
        self.my_logger = logger.my_logger()
        self.wandb_logger = logger.wandb_logger(args)
        
        # Setup hyperparams. args is Namespace object
        # set each attribute as a class attribute
        self.my_logger.info(f'Using Args: {args}\n')
        
        for k, v in vars(args).items():
                setattr(self, k, v)        
    
    def get_n_k(self, key: PRNGKeyArray, bias_val: Optional[int] = None) -> Tuple[Array, Array]:
        n_key, k_key = jax.random.split(key, 2)
        
        rndm_n = get_rand_nums(n_key, 1, self.max_iters, self.batch_size, bias_val)
        rndm_k = get_rand_nums(k_key, jnp.ones(self.batch_size), 
                               self.max_iters - rndm_n + 1, self.batch_size,
                               bias_val)
        
        rndm_k = jnp.ones_like(rndm_n) * self.max_iters
        rndm_n = rndm_k
        
        rndm_n, rndm_k = jnp.clip(rndm_n, 1, self.max_iters), jnp.clip(rndm_k, 1, self.max_iters)
        rndm_n, rndm_k = rndm_n.astype(int), rndm_k.astype(int)
        
        return rndm_n, rndm_k

    def evaluate_acc(self, model: eqx.Module, loader: DataLoader, eval_iters: int, keys: List[PRNGKeyArray]):
        
        metric = []

        for step, batch in tqdm(enumerate(loader), total=len(loader), desc='Validating'):
            batch = batch['text']
            batch = jax.device_put(batch, self.shard)
            
            acc, loss, ppl = self.compute_metrics(model, batch, eval_iters, self.num_classes, keys)
            
            metric.extend([acc, loss, ppl])
        
        # Compute cumulatives
        cum_acc = sum(metric[::3]) / len(metric[::3])
        cum_loss = sum(metric[1::3]) / len(metric[1::3])
        cum_ppl = sum(metric[2::3]) / len(metric[2::3])
        
        return (cum_acc, cum_loss, cum_ppl), batch[0][0] # return one sample for viz
    
    def set_optim_and_scheduler(self, model: eqx.Module):
        assert isinstance(model, eqx.Module), 'Model is not initialized'
        
        total_steps = self.epochs * self.dataset_length // self.batch_size
        
        schedule_fn = optax.warmup_cosine_decay_schedule(self.lr, self.lr, self.warmup_steps,
                                                         total_steps, self.lr // 10)

        # AdamW optimizer with weight decay
        optim = optax.chain(
            optax.clip(self.grad_clip),
            optax.adamw(learning_rate=schedule_fn, weight_decay=self.weight_decay, b1=0.95, b2=0.99)
        )
        
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        
        return optim, opt_state, model
    
    def init_model(self, key: PRNGKeyArray):
        
        if self.baseline:
            model = GPT(self.n_heads, self.seqlen, self.num_blocks, self.width,
                        self.drop_rate, self.num_classes, key)
        else:
            model = React(self.n_heads, self.seqlen, self.max_iters, self.num_blocks, self.width,
                           self.drop_rate, self.num_classes, key)
        
        optim, opt_state, model = self.set_optim_and_scheduler(model)
        count_params(model) # prints to stdout
        
        # switch to half precision
        if self.bf16:
            model = half_precision(model)
        
        return optim, opt_state, model
    
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
    
    @eqx.filter_jit
    def compute_metrics(self, model: eqx.Module, batch: Tuple, eval_iters: int, num_classes: int, keys: List[PRNGKeyArray]):
        '''
        Computes the accuracy, perplexity, loss of the model w.r.t batch
        '''
        input_arr, _, pad_mask = batch
        # make an array of size of batch[0] with each element as eval_iters
        eval_iters = jnp.ones_like(input_arr[:, 0]) * eval_iters
        
        if self.baseline:
            pred_y = jax.vmap(model)(input_arr, pad_mask, keys)
        else:
            pred_y = jax.vmap(model,
                              in_axes=(0, 0, 0, None, None, 0))(
                                  input_arr, eval_iters, pad_mask, False, False, keys)
        
        # compute accuracy
        y_hat = jax.nn.softmax(pred_y, axis=-1).argmax(-1) * pad_mask
        accuracy = jnp.mean(y_hat == batch[1])
        
        # compute loss
        y_one_hot = jax.nn.one_hot(batch[1], num_classes=num_classes) # (batch_size, seqlen, num_classes)
        loss = optax.softmax_cross_entropy(pred_y, y_one_hot).mean()
        
        # compute perplexity
        perplexity = jnp.exp(loss)
        
        return accuracy, loss, perplexity
    
    def train(self, epochs: int, trainloader: DataLoader, valloader: DataLoader, 
              key: PRNGKeyArray):
        
        step_done = 0
        optim, opt_state, model = self.init_model(key)
        print(f'Model: {model}')
        
        if self.resume:
            model, opt_state, epoch_done = self.resume_training(model, opt_state)
        else:
            epoch_done = 0
        
        rndm_n, rndm_k = self.get_n_k(key=key) # initial n and k
        
        for epoch in range(epoch_done, epochs):
            # init empty metrics
            epoch_key = jnp.array([epoch, epoch + 1]).astype(jnp.uint32)
            train_acc, train_loss, train_ppl = [], [], []
            
            keys = jax.random.split(epoch_key, self.batch_size)
            
            for step, batch in tqdm(enumerate(trainloader), total=len(trainloader), desc=f'Epoch {epoch}'):
                # n k bias schedule
                step += step_done # for multiple epochs
                
                batch = batch['text']
                seq, label, pad_mask = jax.device_put(batch, self.shard)
            
                loss, model, opt_state = make_step(model, seq, label, pad_mask, rndm_n, rndm_k,
                                                   optim, opt_state, self.num_classes, keys)
                
                if step % 50 == 0:
                    # cycling through keys to get new n and k
                    #rndm_n, rndm_k = self.get_n_k(key=keys[step % self.batch_size])
                    
                    accuracy, loss, perplexity = self.compute_metrics(model, batch, self.max_iters,
                                                                      self.num_classes, keys)
                    
                    train_acc.append(accuracy)
                    train_loss.append(loss)
                    train_ppl.append(perplexity.mean())
                    
                    loss = loss.item()
                
                    self.wandb_logger.log(
                        {
                            'Train/loss': loss,
                        },
                        step=step
                    )
                                
                if (step + 1) % self.log_interval == 0:
                    # Compute cumulatives
                    cum_train_acc = sum(train_acc) / len(train_acc)
                    cum_train_loss = sum(train_loss) / len(train_loss)
                    cum_train_ppl = sum(train_ppl) / len(train_ppl)
                    
                    ## clear the metrics
                    train_acc, train_loss, train_ppl = [], [], []
                    
                    ## Validation
                    val_metrics, val_sample = self.evaluate_acc(model, valloader, self.max_iters, keys)
                    
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
                    sample_x = seq[0][:9]
                    val_sample_x = val_sample[:9]
                    
                    self.my_logger.info(f"epoch={epoch}, step={step}, loss={loss}")
                    self.my_logger.info(f'Validation accuracy: {val_metrics[0]} | using {self.max_iters} iterations')
                    self.my_logger.info(f'Cumulative Training accuracy: {cum_train_acc}\n')
                    
                    self.generate(model, sample_x, max_new_tokens=96)
                    self.my_logger.info(f'{"=" * 20}\tVal set prompt:\n')
                    self.generate(model, val_sample_x, max_new_tokens=96)
                    
                if step % self.save_interval == 0:
                    # Save the model 
                    filepath = f"{self.save_dir}model_{epoch}_{step}.eqx"
                    
                    save_eqx_obj(self.save_dir, filepath, (model, opt_state))
                    
                    self.wandb_logger.save(filepath)
                        
            print(f'Epoch {epoch} done!')
            step_done = step # prepare for next epoch
                
        return loss, model, opt_state
    
    def generate(self, model: eqx.Module, input_arr: Array, max_new_tokens: int, temperature: float = 0.35):
        '''
        Take a conditioning sequence , call output_head to obtain a prediction
        and autoregressively complete the sequence max_new_tokens times.
        '''
        self.my_logger.info(f'Prompt: {self.decode_fn(input_arr)}')
        inference_model = eqx.tree_inference(model, value=True) # switching to inferencing
        key = jax.random.PRNGKey(0)
        
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
                logits = inference_model(padded_array, pad_mask, key)
                gen = logits[zero_idx - 1, :] / temperature # chose the last token representation
            else:
                logits = inference_model(padded_array, self.max_iters, pad_mask, False, True, key)[1]
                logits = logits[zero_idx - 1, :] # chose the last token representation
                gen = inference_model.out_head(logits) / temperature
                
            # greedy decoding
            gen = gen.argmax()
            input_arr = jnp.concatenate([input_arr, gen.reshape(-1)])
            
        self.my_logger.info(f'model generation: {self.decode_fn(input_arr[-max_new_tokens:-1])}\n')
            
        return input_arr