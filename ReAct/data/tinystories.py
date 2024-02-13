import os
from typing import List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from datasets import load_dataset
from jaxtyping import Array

from .tokenizer import Tok

class TinyStoriesDataset:
    def __init__(self, split: str = 'train', max_length=32, bsz: int = 256, vocab_dir='./ReAct/data'):
        self.bsz = bsz
        self.max_length = max_length + 1
        
        self.dataset = load_dataset('roneneldan/TinyStories', split=split, ignore_verifications=True, 
                                    keep_in_memory=True, num_proc=os.cpu_count())
        
        self.dataset.set_format(type='numpy')
        
        self.tok = Tok(vocab_dir=vocab_dir, max_length=self.max_length)

    def tokenize_and_pad(self, text: List[str]) -> dict[str, List[List[int]]]:
        encoded = self.tok.encode(text['text'])
        
        return {'text': [i.ids for i in encoded]}
    
    def _shift_seq(self, seq: Array) -> Tuple[Array]:
        targets = jnp.roll(seq, shift=-1)
        seq, targets = seq[:, :-1], targets[:, :-1]
        pad_mask = jnp.where(seq != 0, 1, 0)
        
        return seq, targets, pad_mask

    @eqx.filter_jit
    def shift_tokens(self, seq: Array) -> Tuple[Array]:
        return self._shift_seq(seq)

    @staticmethod
    def group_batch(batch: dict) -> dict:
        # Simply batch the data stream
        return {k: [v] for k, v in batch.items()}
    
    def create_dataloader(self, debug: bool = False):
        dataset = self.dataset
        
        if debug or jax.default_backend() == 'cpu':
            samples = 8_000 
            print(f'\nUsing only {samples} samples from the dataset...')
            dataset = dataset.select(range(samples)) # only use some samples
        
        dataset = dataset.map(self.tokenize_and_pad, batched=True, batch_size=self.bsz,
                              keep_in_memory=True, drop_last_batch=True)
        
        dataset = dataset.map(self.group_batch, batched=True, batch_size=self.bsz,
                              keep_in_memory=True, drop_last_batch=True)
        
        return dataset