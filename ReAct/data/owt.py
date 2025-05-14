from typing import Tuple
from .dataset import ParentDataset

class OpenWebTextDataset(ParentDataset):
    def __init__(self, seqlen: int, batch_size: int) -> None:
        super().__init__(
            hf_username='Neel-Gupta',
            hf_dataset='owt',
            tgt_hf_repo='Skylion007/openwebtext',
            max_length=seqlen,
            bsz=batch_size
        )

    def produce_splits(self, split: str, slice: str | None) -> Tuple[str, str]:
        slice = ':99%' if slice is None else slice

        if split == 'train':
            return (split, slice)
        else:
            return ('train', '-1%:') # test split
