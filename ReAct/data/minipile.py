import os
from pathlib import Path
from typing import Dict, List

import jax
import numpy as np
from datasets import load_dataset, load_from_disk
from jaxtyping import Array

from .tokenizer import Tok

class MiniPileDataset:
    def __init__(self, split: str = 'train', max_length: int = 512, bsz: int = 256, vocab_dir: str ='./ReAct/data'):
        self.bsz = bsz
        self.max_length = max_length + 1
        self.split = split

        self.dataset = load_dataset('Neel-Gupta/minipile-processed', split=self.split, ignore_verifications=True,
                                    keep_in_memory=True, num_proc=os.cpu_count() // 2)

        self.dataset.set_format(type='numpy')

        self.tok = Tok(vocab_dir=None, max_length=self.max_length) # vocab_dir is None = GPT2 tokenizer

    def tokenize_and_pad(self, text: List[str]) -> Dict[str, List[List[int]]]:
        encoded = self.tok.encode(text['text'])['input_ids']

        return {'text': encoded}

    @staticmethod
    def shift_tokens(seq: Dict[str, Array]) -> Dict[str, List[Array]]:
        seq = np.asarray(seq['text'])
        
        targets = np.roll(seq, shift=-1)
        seq, targets = seq[:, :-1], targets[:, :-1]
        pad_mask = np.where(seq != 50527 or seq != 50526, 1, 0)

        return {'text': [[seq, targets, pad_mask]]}

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
            chunks += [sentence[i:i + chunk_size] for i in range(0, len(sentence), chunk_size)]

        return {"text": chunks}

    def save_data(self, dataset, path: Path) -> None:
        base_folder = path.parent
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)

        dataset.save_to_disk(
            dataset_path=path,
            num_shards=32 if self.split == 'train' else 8,
            num_proc=os.cpu_count() // 4
        )

        print(f'Saved {self.split} dataset to {path}...')

    def load_data(self, path: Path):
        return load_from_disk(
            dataset_path=path,
            keep_in_memory=True)

    def upload_dataset(self, dataset, hub_path: str = "Neel-Gupta/minipile-processed") -> None:
        '''
        Helper function to upload the dataset to the HuggingFace Hub
        If needed to save on preprocessing wall time.
        '''
        dataset.push_to_hub(hub_path,
                            token=os.getenv('HF_TOKEN'),
                            split=self.split)

    def take_subset(self, dataset) -> None:
        '''
        Take a slice of dataset for debugging purposes
        '''
        if jax.default_backend() == 'cpu':
            samples = 16_000 if self.split == 'train' else 10_000
            print(f'\nUsing only {samples} samples from the dataset...')
            dataset = dataset.select(range(samples)) # only use some samples
        
        return dataset

    def create_dataloader(self):
        dataset = self.dataset
        data_path = Path(f'./cached_data/minipile_{self.split}.data')

        dataset = self.take_subset(dataset)

        if self.bsz == 2048:
            return dataset
        elif os.path.exists(data_path):
            print(f'Loading dataset from {data_path}...')
            dataset = self.load_data(data_path)
            return dataset
        else:
            print(f'Building dataset from scratch... [split: {self.split}] | [bsz: {self.bsz}]')
            
            dataset = load_dataset('JeanKaddour/minipile', split=self.split, ignore_verifications=True,
                                    keep_in_memory=True, num_proc=os.cpu_count() // 2)
            
            dataset = self.take_subset(dataset)
            
            dataset = dataset.map(self.chunk_examples, batched=True, batch_size=self.bsz,
                                keep_in_memory=True, drop_last_batch=True)

            dataset = dataset.map(self.tokenize_and_pad, batched=True, batch_size=self.bsz,
                                keep_in_memory=True, drop_last_batch=True, num_proc=None)

            dataset = dataset.map(self.shift_tokens, batched=True, batch_size=self.bsz,
                                keep_in_memory=True, drop_last_batch=True, num_proc=None)
            
            self.upload_dataset(dataset) # upload the processed dataset to the Hub
            #self.save_data(dataset, data_path) # save the processed dataset locally
            
            return dataset