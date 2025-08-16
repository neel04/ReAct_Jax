from functools import partial
from typing import Callable, Dict, List, Tuple

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset

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
    
    def _process_batch(self, examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Process a batch of examples for streaming dataset"""
        # Chunk examples first
        examples = self.chunk_examples(examples, self.max_length)  # type: ignore

        # Then tokenize and process
        examples = self.tokenize_and_pad(examples, self.tok.encode)  # type: ignore
        examples = self.shift_tokens(examples, self.pad_tok)  # type: ignore

        return examples

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
        # Always use streaming for FinewWeb due to size
        streaming = True
        
        # Load the full streaming dataset
        dataset = load_dataset(
            self.tgt_hf_repo,
            name=self.hf_subset_name,
            split="train",
            verification_mode="no_checks",
            trust_remote_code=True,
            streaming=streaming
        ).select_columns(self.col_name)


        estimated_total_samples = dataset.info.splits["train"].num_examples
        eval_samples = int(estimated_total_samples * 0.01)  # 1% for eval
        
        if split == 'train':
            # Skip the first 1% (eval set) for training
            dataset = dataset.skip(eval_samples)
        elif split in ['val', 'test', 'eval']:
            # Take the first 1% for evaluation
            dataset = dataset.take(eval_samples)
        else:
            raise ValueError(f"Unknown split: {split}")

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

        dataset.with_format(type="numpy") # type: ignore

        print(f"Created streaming {split} dataset from FinewWeb")

        return dataset
