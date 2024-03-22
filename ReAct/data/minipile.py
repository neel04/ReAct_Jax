import os
from typing import List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from datasets import load_dataset
from jaxtyping import Array
from .tokenizer import Tok

class MiniPileDataset:
    def __init__(self, split: str = 'train', max_length: int = 512, bsz: int = 256, vocab_dir: str ='./ReAct/data'):
        self.bsz = bsz
        self.max_length = max_length + 1
        self.split = split
        
        self.dataset = load_dataset('JeanKaddour/minipile', split=self.split, ignore_verifications=True, 
                                    keep_in_memory=True, num_proc=os.cpu_count() // 2)
        
        self.dataset.set_format(type='numpy')
        
        self.tok = Tok(vocab_dir=None, max_length=self.max_length) # vocab_dir is None = GPT2 tokenizer

    def tokenize_and_pad(self, text: List[str]) -> dict[str, List[List[int]]]:
        encoded = self.tok.encode(text['text'])
        
        return {'text': [i.ids for i in encoded]}
    
    @staticmethod
    def shift_tokens(seq: Array) -> Tuple[Array]:
        targets = jnp.roll(seq, shift=-1)
        seq, targets = seq[:, :-1], targets[:, :-1]
        pad_mask = jnp.where(seq != 0, 1, 0)
        
        return seq, targets, pad_mask

    @staticmethod
    def group_batch(batch: dict) -> dict:
        '''
        Simply batch the data stream
        '''
        return {k: [v] for k, v in batch.items()}
    
    def chunk_examples(self, examples: dict) -> dict:
        '''
        Break long sequences into chunks of approx. ctxlen tokens 
        '''
        chunks = []
        chunk_size = (self.max_length - 1) * 4 #rought approx. for ~512 tokens
        
        for sentence in examples["text"]:
            if len(sentence) > chunk_size:
                chunks += [sentence[i:i + chunk_size] for i in range(0, len(sentence), chunk_size)]
            else:
                chunks.append(sentence)
            
        return {"text": chunks}
    
    def create_dataloader(self):
        dataset = self.dataset
        
        if jax.default_backend() == 'cpu':
            samples = 8_000 if self.split == 'train' else 500
            print(f'\nUsing only {samples} samples from the dataset...')
            dataset = dataset.select(range(samples)) # only use some samples
        
        dataset = dataset.map(self.chunk_examples, batched=True, batch_size=self.bsz,
                              keep_in_memory=True, drop_last_batch=True)
        
        dataset = dataset.map(self.tokenize_and_pad, batched=True, batch_size=self.bsz,
                              keep_in_memory=True, drop_last_batch=True)
        
        dataset = dataset.map(self.group_batch, batched=True, batch_size=self.bsz,
                              keep_in_memory=True, drop_last_batch=True)
        
        return dataset