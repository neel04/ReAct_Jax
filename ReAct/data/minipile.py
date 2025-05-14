from typing import Tuple
from .dataset import ParentDataset


class MiniPileDataset(ParentDataset):
    def __init__(self, seqlen: int, batch_size: int) -> None:
        super().__init__(
            hf_username="Neel-Gupta",
            hf_dataset="minipile",
            tgt_hf_repo="JeanKaddour/minipile",
            max_length=seqlen,
            bsz=batch_size,
        )

    def produce_splits(self, split: str, slice: str | None) -> Tuple[str, str]:
        if split == "train":
            return (split, ':100%')
        else:
            return ("test", ":100%")
