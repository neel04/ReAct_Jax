import os
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ReAct.model.react import React
from ReAct.utils.helpers import convert_to_jax, count_params, load_eqx_obj
from ReAct.utils.logger import UnifiedLogger

from .helpers import get_rand_nums


# A unified Trainer class for training and evaluation
@jax.jit
def n_k_loop(model: eqx.Module, input_arr: Array, n: int, k: int) -> Array:
    # forward pass the model without tracking grads
    output, intermediate_array = model(
        jax.lax.stop_gradient(input_arr), 
        iters_to_do=n, prev_thought=None)
    
    # n-k passes, but track the gradient this time
    output, _ = model(input_arr, k, prev_thought=intermediate_array)

    return output

@eqx.filter_value_and_grad
def compute_loss(model: eqx.Module, x: Array, y: Array,
                 n: int, k: int, num_classes: int = 2):
    
    pred_y = jax.vmap(n_k_loop, in_axes=(None, 0, 0, 0))(model, x, n, k) # (batch_size, seqlen, num_classes)
    
    y_one_hot = jax.nn.one_hot(y, num_classes=num_classes) # (batch_size, seqlen, num_classes)
    loss = optax.sigmoid_binary_cross_entropy(pred_y, y_one_hot)
    
    return jnp.mean(loss)
    
@eqx.filter_jit
def make_step(model: eqx.Module, x: Array, y: Array, 
              n: int, k: int, optim, opt_state, num_classes: int = 2):
    
    loss, grads = compute_loss(model, x, y, n, k, num_classes)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

class Trainer:
    def __init__(self, args: dict, key: PRNGKeyArray, logger=None):
        self.key = key
        
        logger = UnifiedLogger(args, level='DEBUG')
        self.my_logger = logger.my_logger()
        self.wandb_logger = logger.wandb_logger(args)
        
        # Setup hyperparams. args is Namespace object
        # set each attribute as a class attribute
        self.my_logger.info(f'Using Args: {args}\n')
        
        for k, v in vars(args).items():
                setattr(self, k, v)        
    
    def get_n_k(self) -> Tuple[Array, Array]:
        n_key, k_key = jax.random.split(self.key, 2)
        
        rndm_n = get_rand_nums(n_key, 0, self.max_iters, self.batch_size)
        rndm_k = get_rand_nums(k_key, jnp.ones(self.batch_size), 
                               self.max_iters - rndm_n + 1, self.batch_size)
        
        return rndm_n, rndm_k

    def evaluate_acc(self, model: eqx.Module, loader: DataLoader, eval_iters: int):
        metric = []

        for step, (x, y) in enumerate(loader):
            x, y = convert_to_jax(x), convert_to_jax(y)
            pred_y = jax.vmap(model, in_axes=(0, None, None, None))(
                x, eval_iters, None, False)
            
            val_pred_y = jax.nn.softmax(pred_y, axis=-1).argmax(-1)
            accuracy = jnp.mean(val_pred_y == y)
            metric.append(accuracy)
        
        return sum(metric) / len(metric), x[0] # return one sample for viz
    
    def set_optim_and_scheduler(self, model: eqx.Module):
        assert isinstance(model, eqx.Module), 'Model is not initialized'
        
        total_steps = self.epochs * self.dataset_length // self.batch_size
        
        schedule_fn = optax.warmup_cosine_decay_schedule(self.lr, self.lr * 2, self.warmup_steps,
                                                         total_steps, self.lr // 20)

        # AdamW optimizer with weight decay
        optim = optax.chain(
            optax.clip(self.grad_clip),
            optax.adamw(learning_rate=schedule_fn, weight_decay=self.weight_decay)
        )
        
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        
        return optim, opt_state, model
    
    def init_model(self, key: PRNGKeyArray):
        # Initialize the model
        model = React(self.seqlen, self.max_iters, self.num_blocks, self.width,
                           self.drop_rate, self.num_classes, key)
        
        optim, opt_state, model = self.set_optim_and_scheduler(model)
        count_params(model)
        
        return model, optim, opt_state
    
    def save_eqx_obj(self, filename: str, obj: eqx.Module):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        with open(filename, "wb") as f:
            eqx.tree_serialise_leaves(f, obj)
    
    def resume_training(self, model: eqx.Module, opt_state: eqx.Module):
        # extracting out the paths
        run_path, epoch = self.resume.split('+')
        run_path, epoch = run_path.strip(), int(epoch.strip())
        
        base_path = "https://api.wandb.ai/files/"
        model_path = f'{base_path}{run_path}/model_{epoch}.eqx'
        opt_state_path = f'{base_path}{run_path}/opt_state_1_{epoch}.eqx'
        
        # wget both files to ReAct/outputs/, if they those files don't exist
        if not os.path.exists(f'{self.save_dir}model_{epoch}.eqx'):
            os.system(f'wget -O {self.save_dir}model_{epoch}.eqx {model_path}')
            os.system(f'wget -O {self.save_dir}opt_state_1_{epoch}.eqx {opt_state_path}')
        
        # load the model and opt_state
        state_0, state_1 = opt_state
        
        model = load_eqx_obj(model, f'{self.save_dir}model_{epoch}.eqx')
        state_1 = load_eqx_obj(state_1, f'{self.save_dir}opt_state_1_{epoch}.eqx')
        
        opt_state = (state_0, state_1)
        
        self.my_logger.info(f'-------- Resuming training from epoch {epoch} ---------\n')
        
        return model, opt_state, epoch
    
    def train(self, epochs: int, trainloader: DataLoader, truncloader: DataLoader,
              valloader: DataLoader, testloader: DataLoader):
        
        model, optim, opt_state = self.init_model(self.key)
        
        if self.resume:
            model, opt_state, epoch_done = self.resume_training(model, opt_state)
        else:
            epoch_done = 0
        
        for epoch in range(epoch_done, epochs):
            rndm_n, rndm_k = self.get_n_k()
            
            for step, (x, y) in tqdm(enumerate(trainloader), total=self.dataset_length // self.batch_size):
                x, y = convert_to_jax(x), convert_to_jax(y)
                
                loss, model, opt_state = make_step(model, x, y, rndm_n, rndm_k,
                                                   optim, opt_state, self.num_classes)
                
                loss = loss.item()
            
            # Validation
            val_acc, val_sample = self.evaluate_acc(model, valloader, self.max_iters)
            val_acc_5, _ = self.evaluate_acc(model, valloader, self.max_iters + 5)
            train_acc, _ = self.evaluate_acc(model, truncloader, self.max_iters)
            test_acc, _ = self.evaluate_acc(model, testloader, self.max_iters + 5)
            
            # Visualize one sample and model prediction
            sample_x = x[0] # Trainng sample
            model_prediction = model(sample_x, self.max_iters, None, False)
            
            val_pred = model(val_sample, self.max_iters, None, False)
            val_pred_5 = model(val_sample, self.max_iters + 5, None, False)
            
            self.wandb_logger.log(
                {
                    'loss': loss,
                    f'Val/acc_{self.max_iters}': val_acc,
                    f'Val/acc_{self.max_iters + 5}': val_acc_5,
                    'Train/acc': train_acc,
                    'Test/acc': test_acc
                },
                
                step=epoch
            )
            
            self.my_logger.info(f"epoch={epoch}, step={step}, loss={loss}")
            self.my_logger.info(f"Sample x:, {sample_x}")
            self.my_logger.info(f"Model prediction:: {model_prediction.argmax(-1)}")
            self.my_logger.info(f'Validation accuracy: {val_acc} | using {self.max_iters} iterations')
            self.my_logger.info(f'Validation accuracy: {val_acc_5} | using {self.max_iters + 5} iterations')
            self.my_logger.info(f'Training accuracy: {train_acc}\n')
            self.my_logger.info(f'Test accuracy: {test_acc}\n')
            
            self.my_logger.info(f'Sample val x: {val_sample}')
            self.my_logger.info(f'Val prediction:\t\t\t{val_pred.argmax(-1)}')
            self.my_logger.info(f'Val prediction @ +5 iters: {val_pred_5.argmax(-1)}')
            self.my_logger.info(f'Correct answer:\t\t\t{val_sample[::-1]}')
            
            complexity = trainloader.dataset.complexity
            
            # Curriculum learning: increase complexity if training accuracy is high
            if epoch > 1 and train_acc >= 0.98 and complexity <= self.cl_seqlen:
                trainloader.dataset.complexity += 1
                print(f'\n\n~~~ New CL dataset complexity: {trainloader.dataset.complexity} ~~~\n\n')
            
            if epoch % self.save_interval == 0:
                # Save the model 
                filepath = f"{self.save_dir}model_{epoch}.eqx"
                self.save_eqx_obj(filepath, model)
                
                # Save the optimizer states
                for i, state in enumerate(opt_state):
                    opt_filepath = f"{self.save_dir}opt_state_{i}_{epoch}.eqx"
                    self.save_eqx_obj(opt_filepath, state)
                    self.wandb_logger.save(opt_filepath, base_path=self.save_dir)
                
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