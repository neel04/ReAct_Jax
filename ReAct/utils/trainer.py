import os
from functools import partial
from typing import Any, Tuple, List, Callable, Optional
from ReAct.model.blocks import MLP, LinearProj, LiteAttention, NewGELU

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ReAct.model.react import React
from ReAct.utils.helpers import convert_to_jax, count_params, load_eqx_obj, save_eqx_obj
from ReAct.utils.logger import UnifiedLogger

from .helpers import get_rand_nums

class DebugReact(eqx.Module):
    '''
    Bare bones model that has a couple of layers
    '''
    max_iters: int = eqx.field(static=True)
    bottleneck: int = eqx.field(static=True)
    SEQLEN: int = eqx.field(static=True)

    proj_1: LinearProj
    act_1: NewGELU
    embed_layer: eqx.nn.Embedding
    pos_enc: jax.Array

    def __init__(self, seqlen: int, max_iters: int, num_blocks: int, width: int,
                 drop_rate: float, tgt_vocab_size: int, key: PRNGKeyArray):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        
        self.bottleneck = width
        self.SEQLEN = seqlen
        self.max_iters = max_iters
        
        self.embed_layer = eqx.nn.Embedding(tgt_vocab_size, self.bottleneck, key=key1)
        self.proj_1 = LinearProj(self.bottleneck, tgt_vocab_size, key=key2)
        self.act_1 = NewGELU()

        self.pos_enc = jax.lax.stop_gradient(self.positional_encoding(self.SEQLEN, self.bottleneck))
    
    @partial(jax.jit, static_argnums=[1,2])
    def positional_encoding(self, seq_len, d_model):
        '''
        Generates the positional encoding for the input sequence
        of shape (batch_size, max_seq_len, d_model) which would be added
        to the sequence embeddings.
        '''
        position = jnp.arange(seq_len, dtype=jnp.float32).reshape(-1, 1)
        div_term = jnp.exp(jnp.arange(0, d_model, 2, dtype=jnp.float32) * -(jnp.log(10000.0) / d_model))
        pe = jnp.zeros((seq_len, d_model))

        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        return pe
    
    @partial(jax.jit, static_argnames=['prev_thought', 'training'])
    def __call__(self, input: Array, attn_mask: Array, iters_to_do: int, 
                 prev_thought: Optional[Array] = None, training: bool = True,
                 key: Optional[PRNGKeyArray] = None) -> Array:
        
        x = self.embed_layer(input) + self.pos_enc # (seqlen, embed_dim)
        x = self.proj_1(self.act_1(x)) # (seqlen, tgt_vocab_size)
        output = jnp.mean(x, axis=0) # (tgt_vocab_size)
        
        if training:
            return output.squeeze(), jnp.zeros_like(output)
        else:
            return output.squeeze()
    
# A unified Trainer class for training and evaluation
@jax.jit
def n_k_loop(model: eqx.Module, input_arr: Array, attn_mask: Array, n: int, k: int, key: PRNGKeyArray) -> Array:
    '''
    # forward pass the model without tracking grads
    output, intermediate_array = model(
        jax.lax.stop_gradient(input_arr), attn_mask,
        iters_to_do=n, prev_thought=None, key=key)
    
    # n-k passes, but track the gradient this time
    output, _ = model(input_arr, attn_mask, k, prev_thought=intermediate_array, key=key)
    '''
    output, _ = model(input_arr, attn_mask, n+k, prev_thought=None, key=key)

    return output

@eqx.filter_value_and_grad
def compute_loss(model: eqx.Module, x: Array, y: Array, attn_mask: Array,
                 n: int, k: int, num_classes: int = 2, keys: PRNGKeyArray = None):
    
    pred_y = jax.vmap(n_k_loop, in_axes=(None, 0, 0, 0, 0, 0))(model, x, attn_mask, n, k, keys) # (batch_size, seqlen, num_classes)
    
    y_one_hot = jax.nn.one_hot(y, num_classes=num_classes) # (batch_size, seqlen, num_classes)
    
    # Softmax cross entropy loss
    loss = optax.softmax_cross_entropy(pred_y, y_one_hot)
    
    return jnp.mean(loss) # across all the baches
    
@eqx.filter_jit
def make_step(model: eqx.Module, x: Array, y: Array, attn_mask: Array, n: int, k: int,
              optim, opt_state, num_classes: int, keys: List[PRNGKeyArray]):  # noqa: F821
    
    loss, grads = compute_loss(model, x, y, attn_mask, n, k, num_classes, keys)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

class Trainer:
    def __init__(self, args: dict, logger=None, shard: Any = None):
        self.shard = shard if shard is not None else None
        self.dataset_length = 2119719
        
        logger = UnifiedLogger(args, level='DEBUG')
        self.my_logger = logger.my_logger()
        self.wandb_logger = logger.wandb_logger(args)
        
        # Setup hyperparams. args is Namespace object
        # set each attribute as a class attribute
        self.my_logger.info(f'Using Args: {args}\n')
        
        for k, v in vars(args).items():
                setattr(self, k, v)        
    
    def get_n_k(self, key: PRNGKeyArray) -> Tuple[Array, Array]:
        n_key, k_key = jax.random.split(key, 2)
        
        rndm_n = get_rand_nums(n_key, 0, self.max_iters, self.batch_size)
        rndm_k = get_rand_nums(k_key, jnp.ones(self.batch_size), 
                               self.max_iters - rndm_n + 1, self.batch_size)
        
        return rndm_n, rndm_k

    def evaluate_acc(self, model: eqx.Module, loader: DataLoader, eval_iters: int,
                     keys: List[PRNGKeyArray], mask_fn: Callable):
        
        metric = []

        for step, batch in enumerate(loader):
            batch = mask_fn(batch)
            seq, label, attn_mask = convert_to_jax(batch)
            seq, label, attn_mask = jax.device_put((seq, label, attn_mask), self.shard)
            
            acc, loss, ppl = self.compute_metrics(model, (seq, label, attn_mask), eval_iters, keys)
            
            metric.extend([acc, loss, ppl])
        
        # Compute cumulatives
        cum_acc = sum(metric[::3]) / len(metric[::3])
        cum_loss = sum(metric[1::3]) / len(metric[1::3])
        cum_ppl = sum(metric[2::3]) / len(metric[2::3])
        
        return (cum_acc, cum_loss, cum_ppl), seq[0] # return one sample for viz
    
    def set_optim_and_scheduler(self, model: eqx.Module):
        assert isinstance(model, eqx.Module), 'Model is not initialized'
        
        total_steps = self.epochs * self.dataset_length // self.batch_size
        
        schedule_fn = optax.warmup_cosine_decay_schedule(self.lr, self.lr * 2, self.warmup_steps,
                                                         total_steps, self.lr // 20)

        # AdamW optimizer with weight decay
        optim = optax.chain(
            optax.clip(self.grad_clip),
            optax.adamw(learning_rate=schedule_fn, weight_decay=self.weight_decay, b1=0.95, b2=0.99)
        )
        
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        
        return optim, opt_state, model
    
    def init_model(self, key: PRNGKeyArray):
        # Initialize the model
        #model = React(self.seqlen, self.max_iters, self.num_blocks, self.width,
        #                   self.drop_rate, self.num_classes, key)
        model = DebugReact(self.seqlen, self.max_iters, self.num_blocks, self.width,
                           self.drop_rate, self.num_classes, key)
        
        optim, opt_state, model = self.set_optim_and_scheduler(model)
        count_params(model)
        
        return optim, opt_state, model
    
    def resume_training(self, model: eqx.Module, opt_state: eqx.Module):
        # extracting out the paths
        run_path, epoch = self.resume.split('+')
        run_path, epoch = run_path.strip(), int(epoch.strip())
        
        base_path = "https://api.wandb.ai/files/"
        model_path = f'{base_path}{run_path}/model_{epoch}.eqx'
        
        # wget both files to ReAct/outputs/, if they those files don't exist
        if not os.path.exists(f'{self.save_dir}model_{epoch}.eqx'):
            os.system(f'wget -O {self.save_dir}model_{epoch}.eqx {model_path}')
        
        model, opt_state = load_eqx_obj(f'{self.save_dir}model_{epoch}.eqx', (model, opt_state))
        
        self.my_logger.info(f'-------- Resuming training from epoch {epoch} ---------\n')
        
        return model, opt_state, epoch
    
    def compute_metrics(self, model: eqx.Module, batch: Tuple, eval_iters: int, keys: List[PRNGKeyArray]):
        '''
        Computes the accuracy, perplexity, loss of the model w.r.t batch
        '''
        pred_y = jax.vmap(model, in_axes=(0, 0, None, None, None, 0))(
            batch[0], batch[2], eval_iters, None, False, keys)
        
        # compute accuracy
        y_hat = jax.nn.softmax(pred_y, axis=-1).argmax(-1)
        accuracy = jnp.mean(y_hat == batch[1])
        
        # compute loss
        y_one_hot = jax.nn.one_hot(batch[1], num_classes=self.num_classes) # (batch_size, seqlen, num_classes)
        loss = optax.softmax_cross_entropy(pred_y, y_one_hot).mean()
        
        # compute perplexity
        perplexity = 2 ** loss
        
        return accuracy, loss, perplexity     
    
    def train(self, epochs: int, trainloader: DataLoader, valloader: DataLoader, 
              key: PRNGKeyArray, mask_fn: Callable, decode_fn: Callable):
        
        optim, opt_state, model = self.init_model(key)
        print(f'Model: {model}')
        
        if self.resume:
            model, opt_state, epoch_done = self.resume_training(model, opt_state)
        else:
            epoch_done = 0
        
        for epoch in range(epoch_done, epochs):
            # init empty metrics
            train_acc, train_loss, train_ppl = [], [], []
            
            keys = jax.random.split(
                jnp.array([epoch, epoch + 1]).astype(jnp.uint32),
                self.batch_size)
            
            rndm_n, rndm_k = self.get_n_k(
                key=jnp.array([epoch, epoch + 1]).astype(jnp.uint32),
                )
            
            for step, batch in tqdm(enumerate(trainloader), total=self.dataset_length // self.batch_size):
                # TODO: remove this hardcoded input
                batch = [jnp.arange(4, 36).tolist()] * self.batch_size
                batch = mask_fn(batch)
                seq, label, attn_mask = convert_to_jax(batch)
                seq, label, attn_mask = jax.device_put((seq, label, attn_mask), self.shard)
                
                loss, model, opt_state = make_step(model, seq, label, attn_mask, rndm_n, rndm_k,
                                                   optim, opt_state, self.num_classes, keys)
                
                accuracy, loss, perplexity = self.compute_metrics(model, (seq, label, attn_mask),
                                                                  self.max_iters, keys)
                
                train_acc.append(accuracy)
                train_loss.append(loss)
                train_ppl.append(perplexity)
                
                loss = loss.item()
                
                if step % 20 == 0:
                    self.wandb_logger.log(
                        {
                            'Train/loss': loss,
                        },
                        step=step
                    )
                                
                if step % self.log_interval == 0:
                    # Comput cumulatives
                    cum_train_acc = sum(train_acc) / len(train_acc)
                    cum_train_loss = sum(train_loss) / len(train_loss)
                    cum_train_ppl = sum(train_ppl) / len(train_ppl)
                    
                    # clear the metrics
                    train_acc, train_loss, train_ppl = [], [], []
                    
                    # Validation
                    val_metrics, val_sample = self.evaluate_acc(model, valloader, self.max_iters, keys, mask_fn)
                    val_metrics_5, _ = self.evaluate_acc(model, valloader, self.max_iters + 5, keys, mask_fn)
                    
                    # Visualize one sample and model prediction
                    viz_key = keys[0]
                    sample_x = seq[0] # Trainng sample
                    attn_mask = jnp.ones_like(sample_x)
                    
                    model_prediction = model(sample_x, attn_mask, self.max_iters, None, False, viz_key)
                    
                    val_pred = model(val_sample, attn_mask, self.max_iters, None, False, viz_key)
                    val_pred_5 = model(val_sample, attn_mask, self.max_iters + 5, None, False, viz_key)
                    
                    self.wandb_logger.log(
                        {
                            f'Train/acc_{self.max_iters}': cum_train_acc,
                            f'Train/cum_loss_{self.max_iters}': cum_train_loss,
                            f'Train/ppl_{self.max_iters}': cum_train_ppl,
                            f'Val/acc_{self.max_iters + 5}': val_metrics_5[0],
                            f'Val/loss_{self.max_iters + 5}': val_metrics_5[1],
                            f'Val/ppl_{self.max_iters + 5}': val_metrics_5[2],
                            f'Val/acc_{self.max_iters}': val_metrics[0],
                            f'Val/loss_{self.max_iters}': val_metrics[1],
                            f'Val/ppl_{self.max_iters}': val_metrics[2],
                        },
                        
                        step=step
                    )
                    
                    self.my_logger.info(f"epoch={epoch}, step={step}, loss={loss}")
                    self.my_logger.info(f"Sample x: {decode_fn(sample_x)} | label: {label[0]} OR {decode_fn(label[0])}")
                    self.my_logger.info(f"Model prediction: {decode_fn([model_prediction.argmax(-1)])}")
                    self.my_logger.info(f'Validation accuracy: {val_metrics[0]} | using {self.max_iters} iterations')
                    self.my_logger.info(f'Validation accuracy: {val_metrics_5[0]} | using {self.max_iters + 5} iterations')
                    self.my_logger.info(f'Cumulative Training accuracy: {cum_train_acc}\n')
                    
                    self.my_logger.info(f'Sample val x: {decode_fn(val_sample)}')
                    self.my_logger.info(f'Val prediction:{decode_fn([val_pred.argmax(-1)])}')
                    self.my_logger.info(f'Val prediction @ +5 iters: {decode_fn([val_pred_5.argmax(-1)])}')
                    
                    if step % self.save_interval == 0:
                        # Save the model 
                        filepath = f"{self.save_dir}model_{epoch}.eqx"
                        
                        save_eqx_obj(self.save_dir, filepath, (model, opt_state))
                        
                        self.wandb_logger.save(filepath)
                
        return loss, model, opt_state

if __name__ == '__main__':
    x = jnp.ones((32)).astype(int)
    y = jnp.ones((32)).astype(int)
    y_one_hot = jax.nn.one_hot(y, num_classes=2)
    
    key = jax.random.PRNGKey(0)
    
    model = React(32, 15, 3, 128, 0.1, 2, key=key)
    model = n_k_loop(model)
    
    @eqx.filter_value_and_grad
    def loss(model, x, y, n, k):
        pred_y = jax.vmap(model)(x, n, k)
        return -jnp.sum(jax.nn.log_softmax(pred_y) * y_one_hot, axis=-1).mean()
    
    output, grads = loss(model, x, y, 10, 5)
    print(output, grads)