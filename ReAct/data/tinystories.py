import os
from functools import partial
from pathlib import Path
from typing import Dict, List

import datasets
import jax
import numpy as np
from datasets import load_dataset, load_from_disk
from jaxtyping import Array

from .tokenizer import Tok

class TinyStoriesDataset:
    def __init__(self, split: str = 'train', max_length: int = 512, bsz: int = 256, vocab_dir: str ='./ReAct/data'):    
        datasets.config.IN_MEMORY_MAX_SIZE = 1e+11
        
        self.cpus = jax.devices("cpu")
        self.pad_tok = 50257
        self.bsz = bsz
        self.max_length = max_length + 1
        self.split = split if split != 'test' else 'validation'

        self.tok = Tok(vocab_dir=None, max_length=self.max_length) # vocab_dir is None = GPT2 tokenizer

    @staticmethod
    def tokenize_and_pad(text: List[str], encode_fn: callable) -> Dict[str, List[List[int]]]:
        encoded = encode_fn(text['text'])['input_ids']

        return {'text': encoded}

    @staticmethod
    def shift_tokens(seq: Dict[str, Array], pad_tok: int) -> Dict[str, List[Array]]:
        seq = np.asarray(seq['text'])
        
        targets = np.roll(seq, shift=-1)
        seq, targets = seq[:, :-1], targets[:, :-1]
        pad_mask = np.where(seq != pad_tok, 1, 0)

        return {'text': [[seq, targets, pad_mask]]}

    @staticmethod
    def chunk_examples(examples: Dict[str, str], max_length: int) -> dict[str, List[str]]:
        '''
        Break long sequences into chunks of approx. ctxlen tokens
        '''
        chunks = []
        chunk_size = (max_length - 1) * 4 #rough approx. for ~512 tokens

        for sentence in examples["text"]:
            chunks += [sentence[i:i + chunk_size] for i in range(0, len(sentence), chunk_size)]

        return {"text": chunks}

    @staticmethod
    def collate_fn(
        examples: Dict[str, List[str]], batch_size: int
    ) -> List[Dict[str, List[str]]]:
        """
        We get examples, a dict with just 1 key - "text", and the value is a list of strings
        Split it into a nested list, where each list has less than self.bsz strings
        """
        examples = examples["text"]
        collated = []

        for i in range(0, len(examples), batch_size):
            collated.append(examples[i : i + batch_size])

        return {"text": collated}

    @staticmethod
    def process_pipeline(
        examples: dict, max_length: int, encode_fn: callable, pad_tok: int, bsz: int
    ) -> Dict[str, List[Array]]:
        """
        Pipeline for processing examples
        """
        examples = TinyStoriesDataset.tokenize_and_pad(examples, encode_fn)
        examples = TinyStoriesDataset.shift_tokens(examples, pad_tok)

        return examples

    def save_data(self, dataset: datasets.Dataset, path: Path) -> None:
        base_folder = path.parent
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)

        dataset.save_to_disk(
            dataset_path=path,
            num_shards=32 if self.split == "train" else 8,
            num_proc=os.cpu_count() // 4,
        )

        print(f'Saved {self.split} dataset to {path}...')

    def load_data(self, path: Path):
        return load_from_disk(dataset_path=path, keep_in_memory=True)

    def upload_dataset(self, dataset: datasets.Dataset, hub_path: str = "Neel-Gupta/tinystories-processed") -> None:
        '''
        Helper function to upload the dataset to the HuggingFace Hub
        If needed to save on preprocessing wall time.
        '''
        dataset.push_to_hub(hub_path,
                            token=os.getenv('HF_TOKEN'),
                            split=self.split)
    
    def take_subset(self, dataset: datasets.Dataset, elements: int) -> None:
        '''
        Take a slice of dataset for debugging purposes
        '''
        if jax.default_backend() == 'cpu':
            samples = elements if self.split == 'train' else elements // 4
            print(f'\nUsing only {samples} samples from the dataset...')
            dataset = dataset.select(range(samples)) # only use some samples
        
        return dataset

    def create_dataloader(self, slice: str = ':99%'):
        data_path = Path(f'./cached_data/tinystories_{self.split}.data')

        try:
            dataset = load_dataset(
                f"Neel-Gupta/tinystories-processed_{self.bsz}",
                split=f"{self.split}[{slice}]",
                verification_mode="no_checks",
                keep_in_memory=False,
                num_proc=None,
            )

            print(f'Loaded {self.split} dataset from HuggingFace Hub')
            
            dataset.set_format(type='numpy')
            
            return dataset
        
        except (FileNotFoundError, ValueError):
            if os.path.exists(data_path):
                print(f'Loading dataset from {data_path}...')
                dataset = self.load_data(data_path)
                return dataset
            else:
                print(f'Building dataset from scratch... [split: {self.split}] | [bsz: {self.bsz}]')

                dataset = load_dataset(
                    "roneneldan/TinyStories",
                    split=f"{self.split}",
                    verification_mode='no_checks',
                    trust_remote_code=True,
                    keep_in_memory=False,
                    num_proc=None,
                ).select_columns('text')

                dataset = self.take_subset(dataset, 2_000)

                def dataset_map_fn(func: callable) -> datasets.dataset_dict:
                    return dataset.map(
                        func,
                        batched=True,
                        batch_size=self.bsz,
                        keep_in_memory=False,
                        drop_last_batch=True,
                        num_proc=None,
                    )
                
                dataset = dataset_map_fn(
                    partial(
                        self.chunk_examples,
                        max_length=self.max_length
                    )
                )
                
                dataset = dataset_map_fn(
                    partial(
                        self.process_pipeline,
                        max_length=self.max_length,
                        encode_fn=self.tok.encode,
                        pad_tok=self.pad_tok,
                        bsz=self.bsz,
                    )
                )

                dataset.set_format(type='numpy')

                self.upload_dataset(dataset, hub_path=f"Neel-Gupta/tinystories-processed_{self.bsz}")

                return dataset
