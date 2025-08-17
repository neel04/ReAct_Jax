from functools import partial
from typing import Callable

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset

from ReAct.utils.helpers import IterableDatasetWithLen

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

    def create_dataloader(
        self,
        split: str,
        slice: str | None = None,
        upload_to_hub: bool = False,
        streaming: bool = True,
    ):
        """
        Override parent method to use streaming=True and implement train/test split
        using take/skip for the massive FinewWeb dataset (~100B tokens).

        For eval, we use ~1% of data (~1B tokens) which should be sufficient.
        """
        dataset = load_dataset(
            self.tgt_hf_repo,
            name=self.hf_subset_name,
            split="train",
            verification_mode="no_checks",
            trust_remote_code=True,
            streaming=True
        ).select_columns(self.col_name)

        def dataset_map_fn(func: Callable) -> Dataset | DatasetDict:
            return dataset.map( # type: ignore
                func,
                batched=True,
                batch_size=self.bsz,
                drop_last_batch=True,
            )

        dataset = dataset_map_fn(
            partial(self.chunk_examples, max_length=self.max_length)
        )

        dataset = dataset_map_fn(
            partial(
                self.process_pipeline,
                encode_fn=self.tok.encode,
                pad_tok=self.pad_tok,
            )
        )

        total_batches = dataset.info.splits["train"].num_examples // self.bsz
        eval_samples = int(total_batches * 0.01)  # 1% for eval

        if split == "train":
            dataset = dataset.skip(eval_samples)
            _length = (total_batches - eval_samples)
        elif split in ["val", "test", "eval"]:
            dataset = dataset.take(eval_samples)  # take first 1%
            _length = eval_samples
        else:
            raise ValueError(f"Unknown split: {split}")

        if slice:
            dataset = dataset.take(
                taken_samples := int(( int(slice[1:-1])) / 100 * _length)
            )
            _length = taken_samples

        dataset.with_format(type="numpy") # type: ignore

        print(f"Created streaming {split} dataset from FinewWeb")

        return IterableDatasetWithLen(dataset, _length)
