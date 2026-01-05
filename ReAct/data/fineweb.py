import os
from functools import partial
from pathlib import Path
from typing import Callable

import jax
import torch.multiprocessing as multiprocessing
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.iterable_dataset import IterableDataset
from datasets.load import load_dataset
from huggingface_hub import snapshot_download

from ReAct.utils.helpers import IterableDatasetWithLen, temp_cwd

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

    def map_factory(self, dataset):
        def dataset_map_fn(func: Callable) -> IterableDataset:
            return dataset.map(  # type: ignore
                func,
                batched=True,
                batch_size=self.bsz,
                drop_last_batch=True,
            )

        return dataset_map_fn

    def predownload_dataset(self) -> None:
        disk_path = os.getenv("DISK_PATH")

        if not disk_path:
            raise EnvironmentError("DISK_PATH is not set.")

        base_dir = Path(disk_path)
        base_dir.mkdir(parents=True, exist_ok=True)

        with temp_cwd(base_dir):
            snapshot_download(
                repo_id="HuggingFaceFW/fineweb",
                repo_type="dataset",
                local_dir="./fineweb-sample-100BT",
                allow_patterns=["sample/100BT/*"],
            )

    def create_dataloader(
        self,
        split: str,
        slice: str | None = None,
        upload_to_hub: bool = False,
        streaming: bool = True,
        start_step: int = 0,
    ):
        """
        Override parent method to use streaming=True and implement train/test split
        using take/skip for the massive FinewWeb dataset (~100B tokens).

        For eval, we use ~1% of data (~1B tokens) which should be sufficient.
        """
        # No need for data preprocessing on non-primary processes
        if jax.process_index() != 0:
            total_batches = 147639585 // self.bsz
            eval_samples = int(total_batches * 0.01)

            if split == "train":
                _length = total_batches - eval_samples
            else:
                _length = eval_samples

            if slice:
                taken_samples = int((int(slice[1:-1])) / 100 * _length)
                _length = taken_samples

            # Pass on a dummy dataset instead
            return IterableDatasetWithLen(
                HFDataset.from_dict({"text": "Dummy dataset :)"}), _length
            )

        disk_path = os.getenv("DISK_PATH")
        local_dir = Path(disk_path) / "fineweb-sample-100BT" if disk_path else None

        if local_dir and not local_dir.exists():
            try:
                self.predownload_dataset()
            except Exception as exc:
                print(f"Predownload failed, falling back to HF streaming: {exc}")

        if local_dir and local_dir.exists():
            dataset: IterableDataset = load_dataset(  # pyright: ignore[reportAssignmentType]
                "parquet",
                data_dir=str(local_dir),
                split="train",
                verification_mode="no_checks",
                streaming=True,
            )
        else:
            dataset = load_dataset(  # pyright: ignore[reportAssignmentType]
                self.tgt_hf_repo,
                name=self.hf_subset_name,
                split="train",
                verification_mode="no_checks",
                trust_remote_code=True,
                streaming=True,
            )

        total_batches = 147639585 // self.bsz
        eval_samples = int(total_batches * 0.01)  # 1% for eval

        dataset = dataset.select_columns(self.col_name)

        dataset_map_fn = self.map_factory(dataset)

        dataset = dataset_map_fn(
            partial(self.chunk_examples, max_length=self.max_length)
        )

        if start_step != 0:
            dataset = dataset.skip(start_step)

        dataset = dataset_map_fn(
            partial(
                self.process_pipeline,
                encode_fn=self.tok.encode,
                pad_tok=self.pad_tok,
            )
        )

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

        dataset = dataset.shuffle(seed=42, buffer_size=2 ** 8)

        return IterableDatasetWithLen(dataset, _length)
