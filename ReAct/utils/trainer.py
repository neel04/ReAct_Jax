from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float16, PRNGKeyArray
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ReAct.model.react import React
from ReAct.utils.helpers import convert_to_jax, count_params
from ReAct.utils.logger import UnifiedLogger

from .helpers import get_rand_nums


# A unified Trainer class for training and evaluation
class n_k_loop(eqx.Module):
    my_model: eqx.Module

    def __init__(self, model: eqx.Module):
        self.my_model = model

    @jax.jit
    def __call__(self, input_arr: Array, n: int, k: int) -> Array:
        # forward pass the model without tracking grads
        output, intermediate_array = self.my_model(jax.lax.stop_gradient(input_arr),
                                                   iters_to_do=n, prev_thought=None)
        # n-k passes, but track the gradient this time
        output, _ = self.my_model(input_arr, k, prev_thought=intermediate_array)
        
        return output

class Trainer:
    def __init__(self, args: dict, key: PRNGKeyArray, logger=None):
        self.key = key
        
        logger = UnifiedLogger(args, level='DEBUG', mode='disabled')
        self.my_logger = logger.my_logger()
        self.wandb_logger = logger.wandb_logger(args)
        
        # Setup hyperparams. args is Namespace object
        # set each attribute as a class attribute
        for k, v in vars(args).items():
            setattr(self, k, v)        
    
    def get_n_k(self) -> Tuple[Array, Array]:
        n_key, k_key = jax.random.split(self.key, 2)
        
        rndm_n = get_rand_nums(n_key, 0, self.max_iters, self.batch_size)
        rndm_k = get_rand_nums(k_key, jnp.ones(self.batch_size), 
                               self.max_iters - rndm_n + 1, self.batch_size)
        
        return rndm_n, rndm_k

    @eqx.filter_value_and_grad
    def compute_loss(self, model: eqx.Module, x: Float16[Array, '...'],
                     y: Float16[Array, '...'], n: int, k: int):
        
        class_weights = jnp.array([0.35, 0.65])
        pred_y = jax.vmap(model)(x, n, k)
        
        y_one_hot = jax.nn.one_hot(y, num_classes=self.num_classes)
        loss = -jnp.sum(jax.nn.log_softmax(pred_y) * y_one_hot * class_weights, axis=-1)
        
        return loss.mean()
    
    @eqx.filter_jit
    def make_step(self, model: eqx.Module, x: Float16[Array, '...'],
                    y: Float16[Array, '...'], n: int, k: int, optim, opt_state):
        
        loss, grads = self.compute_loss(model, x, y, n, k)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    
    def evaluate_acc(self, model: eqx.Module, loader: DataLoader, eval_iters: int):
        metric = []

        for step, (x, y) in enumerate(loader):
            x, y = convert_to_jax(x), convert_to_jax(y)
            pred_y = jax.vmap(model, in_axes=(0, None))(x, eval_iters)
            val_pred_y = jax.nn.softmax(pred_y, axis=-1).argmax(-1)
            accuracy = jnp.mean(val_pred_y == y)
            metric.append(accuracy)

        return sum(metric) / len(metric)
    
    def set_optim_and_scheduler(self, model: eqx.Module):
        assert isinstance(model, eqx.Module), 'Model is not initialized'
        
        total_steps = self.num_epochs * self.dataset_length // self.batch_size
        
        schedule_fn = optax.warmup_cosine_decay_schedule(self.lr, self.lr * 2, self.warmup_steps,
                                                         total_steps, self.lr // 20)

        # AdamW optimizer with weight decay
        optim = optax.chain(
            optax.clip(self.grad_clip),
            optax.adamw(learning_rate=schedule_fn, weight_decay=self.weight_decay)
        )
        
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        
        self.optim = optim
        self.opt_state = optim.init(eqx.filter(model, eqx.is_array))
        
        return optim, opt_state, model
    
    def init_model(self, key: PRNGKeyArray):
        # Initialize the model
        model = React(self.seqlen, self.max_iters, self.num_blocks, self.width,
                           self.drop_rate, self.num_classes, key)
        
        optim, opt_state, model = self.set_optim_and_scheduler(model)
        count_params(model)
        
        return model, optim, opt_state
    
    def save_model(self, filename: str, model: eqx.Module):
        with open(filename, "wb") as f:
            eqx.tree_serialise_leaves(f, model)
        
    def train(self, epochs: int, trainloader: DataLoader,
              truncloader: DataLoader, valloader: DataLoader):
        
        model, optim, opt_state = self.init_model(self.key)
        model = n_k_loop(model)
        
        for epoch in range(epochs):
            rndm_n, rndm_k = self.get_n_k()
            
            for step, (x, y) in tqdm(enumerate(trainloader), total=self.dataset_length // self.batch_size):
                x, y = convert_to_jax(x), convert_to_jax(y)
                
                loss, model, opt_state = self.make_step(model, x, y, rndm_n, rndm_k,
                                                        self.optim, self.opt_state)
                
                loss = loss.item()
            
            # Validation
            val_acc = self.evaluate_acc(model, valloader, self.max_iters)
            val_acc_5 = self.evaluate_acc(model, valloader, self.max_iters + 5)
            train_acc = self.evaluate_acc(model, truncloader, self.max_iters)
            
            # Visualize one sample and model prediction
            sample_x = x[0]
            model_prediction = model(sample_x, self.max_iters)  # Get model prediction for the sample
            
            self.wandb_logger.log(
                {
                    'loss': loss,
                    f'val_acc_{self.max_iters}': val_acc,
                    f'val_acc_{self.max_iters + 5}': val_acc_5,
                    'train_acc': train_acc
                },
                
                step=epoch
            )
            
            self.my_logger.info(f"epoch={epoch}, step={step}, loss={loss}")
            self.my_logger.info(f"Sample x:, {sample_x}")
            self.my_logger.info(f"Model prediction:: {model_prediction.argmax(-1)}")
            self.my_logger.info(f'Validation accuracy: {val_acc} | using {self.max_iters} iterations')
            self.my_logger.info(f'Validation accuracy: {val_acc_5} | using {self.max_iters + 5} iterations')
            self.my_logger.info(f'Training accuracy: {train_acc}')
            
            
            if epoch > 1 and train_acc >= 0.98 and self.dataset_length <= (self.seqlen - 6):
                self.dataset_length += 1
                print(f'\n~~~ New CL dataset complexity: {self.dataset_length} ~~~')
            
            if epoch % self.save_interval == 0:
                # Save the model 
                self.save_model(f'{self.save_dir}model_{epoch}.eqx', model)
                self.wandb_logger.save(f'{self.save_dir}model_{epoch}.eqx')
                
        return loss, model, opt_state