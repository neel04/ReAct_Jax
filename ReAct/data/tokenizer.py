from tokenizers import Tokenizer, normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.processors import TemplateProcessing

class Tok:
    def __init__(self, vocab_dir: str, max_length: int):
        self.vocab_dir = vocab_dir
        self.max_length = max_length
        self.tokenizer = Tokenizer.from_file(f'{self.vocab_dir}/tinytok.json')
        
        self.tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            pair="[SOS] $A [EOS] $B:1 [EOS]:1",
            special_tokens=[
                ("[SOS]", 1),
                ("[EOS]", 2),
            ])
        
        self.tokenizer.normalizer = normalizers.Sequence([
            NFD(),
            Lowercase(),
            StripAccents(),
        ])
        
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=self.max_length)
        self.tokenizer.enable_truncation(max_length=self.max_length)
    
    def encode(self, text: str):
        if len(text) > 1:
            return self.tokenizer.encode_batch(text)
        elif isinstance(text, list):
            return self.tokenizer.encode(text[0])
        else:
            return self.tokenizer.encode(text)
    
    def decode(self, ids: list):
        decoded = self.tokenizer.decode(ids, skip_special_tokens=False)
        return decoded.replace('[UNK]', '').replace('[PAD]', '')
    
    def save(self):
        self.tokenizer.save(f'./ReAct/data/{self.dataset}tok.json')
    
    def load(self):
        self.tokenizer = Tokenizer.from_file(f'./ReAct/data/{self.dataset}tok.json')
        
        self.tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            pair="[SOS] $A [EOS] $B:1 [EOS]:1",
            special_tokens=[
                ("[SOS]", 1),
                ("[EOS]", 2),
                ("[MASK]", 3)
            ])
        
        self.tokenizer.normalizer = normalizers.Sequence([
            NFD(),
            Lowercase(),
            StripAccents(),
        ])
        
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=self.max_length)
        self.tokenizer.enable_truncation(max_length=self.max_length)
    
    def __repr__(self):
        return f'Tok(dataset={self.dataset}, max_length={self.max_length})'
    
    def __str__(self):
        return f'Tok(dataset={self.dataset}, max_length={self.max_length})'
    
    def __call__(self, text: str):
        return self.encode(text)
    
    def __len__(self):
        return self.max_length

if __name__ == '__main__':
    tok = Tok('./ReAct/data/', 32)
    out = tok(['Sam and alice go and stab diana for [MASK]', '[MASK] off'])
    print('Vocab size:', tok.tokenizer.get_vocab_size())
    print(dict(zip(out[0].tokens, out[0].ids)))
    
    # decode [0, 1, 2, 614, 69, 420]
    print(
        tok.decode([0, 1, 2, 101, 69, 420])
    )