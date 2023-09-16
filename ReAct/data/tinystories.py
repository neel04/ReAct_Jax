import os

import torch
from datasets import load_dataset
from tokenizer import Tok
from tqdm import tqdm


class TinyStories:
    def __init__(self, split: str = 'train', max_length=32, bsz: int = 256, vocab_dir='./ReAct/data'):
        self.bsz = bsz
        
        self.dataset = load_dataset('roneneldan/TinyStories', split=split, streaming=True,
                                    ignore_verifications=True, keep_in_memory=False)
        
        self.tok = Tok(vocab_dir=vocab_dir, max_length=max_length)

    def tokenize_and_pad(self, text: str):
        encoded = self.tok.encode(text['text'])
        
        return {'text': [i.ids for i in encoded]}

    def create_dataloader(self):
        self.dataset = self.dataset.map(self.tokenize_and_pad, batched=True, batch_size=self.bsz)
        self.dataset.with_format(type='torch')
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.bsz,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=True)
        
        return self.dataloader

if __name__ == '__main__':
    dataset = TinyStories()
    dataloader = dataset.create_dataloader()
    
    for batch in tqdm(dataloader):
        pass