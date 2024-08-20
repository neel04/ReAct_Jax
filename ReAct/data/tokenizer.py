from typing import Dict, List, Optional, Any

from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

class Tok:
    def __init__(self, vocab_dir: Optional[str], max_length: int):
        self.vocab_dir = vocab_dir
        self.max_length = max_length

        if vocab_dir is not None:
            self.tokenizer = AutoTokenizer.from_file(f"{self.vocab_dir}/tinytok.json")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "gpt2", clean_up_tokenization_spaces=True
            )

        self.tokenizer.post_processor = TemplateProcessing(
            single="$A <|endoftext|>",
            special_tokens=[
                ("<|endoftext|>", 50526),
            ],
        )

        self.tokenizer.normalizer = normalizers.Sequence(
            [
                NFD(),
                Lowercase(),
                StripAccents(),
            ]
        )

        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def encode(self, text: List[str] | str) -> Dict[str, List]:
        return self.tokenizer(
            text, padding="max_length", max_length=self.max_length, truncation=True
        )

    def decode(self, ids: Any) -> str:
        ids = [int(i) for i in ids]  # ensure its integers

        decoded = self.tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )

        return decoded.replace("!", "")

    def save(self):
        self.tokenizer.save(f"./ReAct/data/{self.dataset}tok.json")

    def __repr__(self):
        return f"Tok(dataset={self.dataset}, max_length={self.max_length})"

    def __str__(self):
        return f"Tok(dataset={self.dataset}, max_length={self.max_length})"

    def __call__(self, text: str):
        return self.encode(text)

    def __len__(self):
        return self.max_length
