from typing import Tuple
from .dataset import ParentDataset


class GithubCodeDataset(ParentDataset):
    def __init__(self, seqlen: int, batch_size: int) -> None:
        super().__init__(
            hf_username="Neel-Gupta",
            hf_dataset="ghcode",
            tgt_hf_repo="codeparrot/codeparrot-clean",
            max_length=seqlen,
            bsz=batch_size,
            col_name='content'
        )

    def produce_splits(self, split: str, slice: str | None) -> Tuple[str, str]:
        slice = ":33%" if slice is None else slice

        if split == "train":
            return (split, slice)
        else:
            return ("train", "-1%:")  # test split
