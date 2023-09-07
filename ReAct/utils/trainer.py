import json
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torch.utils.data.dataloader as DataLoader

from helpers import get_rand_nums
from jaxtyping import Array, Float16, PRNGKeyArray
from tqdm import tqdm

from ReAct.model.react import ReAct
from ReAct.utils.helpers import convert_to_jax, count_params


# A unified Trainer class for training and evaluation
class Trainer:
    def __init__(self, args: dict, key: PRNGKeyArray, logger=None):
        self.key = key
        self.my_logger = logger.my_logger('DEBUG')
        self.wandb_logger = logger.wandb_logger()
        
        # Setup hyperparams
        for k, v in args.items():
            setattr(self, k, v)
    
    def get_n_k(self) -> Tuple[Array, Array]:
        n_key, k_key = jax.random.split(self.key, 2)
        
        rndm_n = get_rand_nums(n_key, 0, self.max_iters, self.batch_size)
        rndm_k = get_rand_nums(k_key, jnp.ones(self.batch_size), 
                               self.max_iters - rndm_n + 1, self.batch_size)
        
        return rndm_n, rndm_k

    @jax.jit
    def react_forward(self, input: Float16[Array, '...'],
                      n: int, k: int) -> Float16[Array, '...']:
        
        # forward pass the model without tracking grads
        intermediate_array = self.model(jax.lax.stop_gradient(input), iters_to_do=n, prev_thought=None)
        # n-k passes, but track the gradient this time
        return self.model(input, k, intermediate_array)
    
    @eqx.filter_value_and_grad
    def compute_loss(self, model: eqx.Module, x: Float16[Array, '...'],
                     y: Float16[Array, '...'], n: int, k: int):
        
        pred_y = self.react_forward(x, n, k)
        y_one_hot = jax.nn.one_hot(y, num_classes=2)
        pred_probs = jax.nn.softmax(pred_y, axis=-1)
        loss = -jnp.mean(jnp.sum(y_one_hot * jnp.log(pred_probs), axis=-1))
        return loss
    
    @eqx.filter_jit
    def make_step(self, model: eqx.Module, x: Float16[Array, '...'],
                    y: Float16[Array, '...'], n: int, k: int, optim, opt_state: eqx.OptState):
        
        loss, grads = self.compute_loss(model, x, y, n, k)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    
    def evaluate_acc(self, model: eqx.Module, loader: DataLoader, eval_iters: int):
        metric = []

        for step, (x, y) in enumerate(loader):
            x, y = convert_to_jax(x), convert_to_jax(y)
            pred_y = self.react_forward(x, jnp.zeros(self.batch_size).astype(int), 
                                        jnp.zeros(self.batch_size).astype(int) + eval_iters)
            val_pred_y = jax.nn.softmax(pred_y, axis=-1).argmax(-1)
            accuracy = jnp.mean(val_pred_y == y)
            metric.append(accuracy)

        return sum(metric) / len(metric)
    
    def set_optim_and_scheduler(self):
        assert isinstance(self.model, eqx.Module), 'Model is not initialized'
        
        total_steps = self.num_epochs * self.dataset_length // self.batch_size
        
        schedule_fn = optax.warmup_cosine_decay_schedule(self.lr, self.lr * 2, self.warmup_steps,
                                                         total_steps, self.lr // 20)

        # AdamW optimizer with weight decay
        optim = optax.chain(
            optax.clip(self.grad_clip),
            optax.adamw(learning_rate=schedule_fn, weight_decay=self.weight_decay)
        )
        
        opt_state = optim.init(eqx.filter(self.model, eqx.is_array))
        
        self.optim = optim
        self.opt_state = optim.init(eqx.filter(self.model, eqx.is_array))
        
        return optim, opt_state
    
    def init_model(self, key: PRNGKeyArray):
        # Initialize the model
        self.model = ReAct(self.seqlen, self.max_iters, self.num_blocks, self.width,
                           self.drop_rate, self.num_classes, key)
        
        optim, opt_state = self.set_optim_and_scheduler()
        count_params(self.model)
        
        return self.model, optim, opt_state
    
    @staticmethod
    def save_model(filename: str, model: eqx.Module, hyperparams: dict):
        with open(filename, "wb") as f:
            hyperparam_str = json.dumps(hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, model)
        
    def train(self, epochs: int,
                        trainloader: DataLoader, truncloader: DataLoader, valloader: DataLoader):
        
        self.model, self.optim, self.opt_state = self.init_model(self.key)
        
        for epoch in range(epochs):
            rndm_n, rndm_k = self.get_n_k()
            
            for step, (x, y) in tqdm(enumerate(trainloader), total=self.dataset_length // self.batch_size):
                x, y = convert_to_jax(x), convert_to_jax(y)
                
                loss, model, opt_state = self.make_step(self.model, x, y, rndm_n, rndm_k,
                                                        self.optim, self.opt_state)
                
                loss = loss.item()
            
            # Validation
            val_acc = self.evaluate_acc(model, valloader, self.max_iters)
            val_acc_5 = self.evaluate_acc(model, valloader, self.max_iters + 5)
            train_acc = self.evaluate_acc(model, truncloader, self.max_iters)
            
            # Visualize one sample and model prediction
            sample_x = x[0]  # Select the first sample from the batch
            model_prediction = model(sample_x, 0, self.max_iters)  # Get model prediction for the sample
            
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
            self.my_logger.info("Sample x:", sample_x)
            self.my_logger.info("Model prediction:", model_prediction.argmax(-1))
            self.my_logger.info(f'Validation accuracy: {val_acc} | using {self.max_iters} iterations')
            self.my_logger.info(f'Validation accuracy: {val_acc_5} | using {self.max_iters + 5} iterations')
            self.my_logger.info(f'Training accuracy: {train_acc}')
            
            
            if epoch > 1 and train_acc >= 0.98 and dataset.length <= (self.seqlen - 6):
                dataset.length += 1
                print(f'\n~~~ New CL dataset complexity: {dataset.length} ~~~')
            
            if epoch % self.save_interval == 0:
                # Save the model 
                self.save_model(f'{self.save_dir}model_{epoch}.eqx', model, vars(self))
                self.wandb_logger.save(f'{self.save_dir}model_{epoch}.eqx')
                
            return loss, model, opt_state