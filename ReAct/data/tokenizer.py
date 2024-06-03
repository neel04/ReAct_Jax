from typing import Dict, List, Optional

from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

class Tok:
    def __init__(self, vocab_dir: Optional[str], max_length: int):
        self.vocab_dir = vocab_dir
        self.max_length = max_length
        
        if vocab_dir is not None:
            self.tokenizer = AutoTokenizer.from_file(f'{self.vocab_dir}/tinytok.json')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        
        self.tokenizer.post_processor = TemplateProcessing(
            single="$A <|endoftext|>",
            special_tokens=[
                ("<|endoftext|>", 50526),
            ])
        
        self.tokenizer.normalizer = normalizers.Sequence([
            NFD(),
            Lowercase(),
            StripAccents(),
        ])
        
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    def encode(self, text: List[str]) -> Dict[str, List]:
        return self.tokenizer(text,
                              padding='max_length',
                              max_length=self.max_length,
                              truncation=True)
    
    def decode(self, ids: list):
        # convert ids to a list of ints
        ids = [int(i) for i in ids]
        decoded = self.tokenizer.decode(ids, skip_special_tokens=False)
        
        return decoded.replace('!', '')
    
    def save(self):
        self.tokenizer.save(f'./ReAct/data/{self.dataset}tok.json')
    
    def __repr__(self):
        return f'Tok(dataset={self.dataset}, max_length={self.max_length})'
    
    def __str__(self):
        return f'Tok(dataset={self.dataset}, max_length={self.max_length})'
    
    def __call__(self, text: str):
        return self.encode(text)
    
    def __len__(self):
        return self.max_length

if __name__ == '__main__':
    tok = Tok(None, 32)
    out = tok(['Sam and alice go and stab diana for no good reason (they are pschyopaths)', 'mask off'])
    
    print('Vocab size:', tok.tokenizer.get_vocab_size())
    print(dict(zip(out[0].tokens, out[0].ids)))
    
    print(
        tok.decode([13482, 338, 20854])
    )