from typing import Tuple
from .dataset import ParentDataset


class TinyStoriesDataset(ParentDataset):
    def __init__(self, seqlen: int, batch_size: int) -> None:
        super().__init__(
            hf_username="Neel-Gupta",
            hf_dataset="tinystories",
            tgt_hf_repo="roneneldan/TinyStories",
            max_length=seqlen,
            bsz=batch_size
        )

    def produce_splits(self, split: str, slice: str | None) -> Tuple[str, str]:
        if split == "train":
            return (split, ':100%')
        else:
            return ("validation", ":100%")
