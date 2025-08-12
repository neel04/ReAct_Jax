from typing import Tuple
from .dataset import ParentDataset


class FineWebDataset(ParentDataset):
    def __init__(self, seqlen: int, batch_size: int) -> None:
        super().__init__(
            hf_username="Neel-Gupta",
            hf_dataset="fineweb",
            tgt_hf_repo="HuggingFaceFW/fineweb",
            hf_subset_name="sample-100BT",
            max_length=seqlen,
            bsz=batch_size,
        )

    def produce_splits(self, split: str, slice: str | None) -> Tuple[str, str]:
        """
        FineWeb uses 'train' split for all data. We can use different slices
        to control how much data to use.
        For the 100B sample slice, we use the 'sample-100BT' config.
        """
        slice = ':99%' if slice is None else slice

        if split == 'train':
            return (split, slice)
        else:
            return ('train', '-1%:')  # test split uses last 1%
