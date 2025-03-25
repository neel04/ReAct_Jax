import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, cast

import jax
import numpy as np
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset, load_from_disk
from jaxtyping import Array
from numpy._typing import NDArray

from .tokenizer import Tok


class ParentDataset:
    def __init__(
        self,
        hf_username: str,
        hf_dataset: str,
        tgt_hf_repo: str,
        max_length: int,
        bsz: int,
        col_name: str = "text",
    ) -> None:
        """
        - `hf_username` and `hf_dataset` are derived from the users own HF repo details.
        This is used to store processed datasets as they take really long to process.

        - `tgt_hf_repo` is the actual (unprocessed/raw) dataset stored @ HF.
        """
        self.tok = Tok(vocab_dir=None, max_length=max_length + 1)
        self.pad_tok = 50257

        # User defined attributes
        self.max_length = max_length + 1
        self.hf_username = hf_username
        self.hf_dataset = hf_dataset
        self.tgt_hf_repo = tgt_hf_repo
        self.col_name = col_name
        self.bsz = bsz

    @staticmethod
    def shift_tokens(
        seq: Dict[str, Array], pad_tok: int
    ) -> Dict[str, List[List[NDArray]]]:
        _seq: NDArray = np.asarray(seq["text"])

        targets = np.roll(_seq, shift=-1)
        _seq, targets = _seq[:, :-1], targets[:, :-1]
        pad_mask = np.where(_seq != pad_tok, 1, 0)

        return {"text": [[_seq, targets, pad_mask]]}

    @staticmethod
    def tokenize_and_pad(
        text: dict[str, str], encode_fn: Callable
    ) -> dict[str, list[list[int]]]:
        encoded = encode_fn(text["text"])["input_ids"]
        return {"text": encoded}

    @staticmethod
    def chunk_examples(
        examples: Dict[str, str], max_length: int
    ) -> Dict[str, List[str]]:
        """
        Break long sequences into chunks of approx. ctxlen tokens
        """
        chunks = []
        chunk_size = (max_length - 1) * 4  # rough approx. for ~512 tokens

        for sentence in examples["text"]:
            chunks += [
                sentence[i : i + chunk_size]
                for i in range(0, len(sentence), chunk_size)
            ]

        return {"text": chunks}

    @staticmethod
    def collate_fn(
        examples: Dict[str, List[str]], batch_size: int
    ) -> Dict[str, List[str]]:
        """
        We get examples, a dict with just 1 key - "text", and the value is a list of strings
        Split it into a nested list, where each list has less than bsz strings
        """
        _examples = examples["text"]
        collated = []

        for i in range(0, len(_examples), batch_size):
            collated.append(_examples[i : i + batch_size])

        return {"text": collated}

    @staticmethod
    def load_data(path: Path):
        return load_from_disk(dataset_path=path, keep_in_memory=False)

    @staticmethod
    def save_data(split: str, dataset: Any, path: Path) -> None:
        base_folder = path.parent

        if not os.path.exists(base_folder):
            os.makedirs(base_folder)

        dataset.save_to_disk(
            dataset_path=path,
            num_shards=32 if split == "train" else 8,
            num_proc=os.cpu_count() // 4,  # type: ignore
        )

        print(f"Saved {split} dataset to {path}...")

    @staticmethod
    def upload_dataset(
        split: str,
        dataset: Any,
        hub_path: str = "Neel-Gupta/owt-processed",
        allow_upload: bool = False,
    ) -> None:
        """
        Helper function to upload the dataset to the HuggingFace Hub
        If needed to save on preprocessing wall time.
        """
        if allow_upload:
            dataset.push_to_hub(hub_path, token=os.getenv("HF_TOKEN"), split=split)
            return

        print(
            "\nWARNING: Synthesized dataset not uploaded. Pass in the dataset upload flag to do so.\n"
        )

    @staticmethod
    def take_subset(split: str, dataset: Any, elements: int) -> Dataset:
        """
        Take a slice of dataset for debugging purposes
        """
        if jax.default_backend() == "cpu":
            samples = elements if split == "train" else elements // 4
            print(f"\nUsing only {samples} samples from the dataset...")
            dataset = dataset.select(range(samples))  # only use some samples

        return cast(Dataset, dataset)

    def process_pipeline(
        self, examples: Dict, encode_fn: Callable, pad_tok: int
    ) -> Dict[str, List[Array]]:
        """
        Pipeline for processing examples
        """
        examples = self.tokenize_and_pad(examples, encode_fn)
        examples = self.shift_tokens(examples, pad_tok)

        return examples

    def produce_splits(self, split: str, slice: str | None) -> Tuple[str, str]:
        """
        Returns: A tuple of strings that are the split & slice respectively.
        This is the method you would have to override and change as it depends.
        """
        raise NotImplementedError

    def create_dataloader(
        self, split: str, slice: str | None = None, upload_to_hub: bool = False
    ):
        data_path = Path(f"{os.getenv('DISK_PATH')}/cached_data/owt_{split}.data")

        split, slice = self.produce_splits(split, slice)

        try:
            print(f"Loading dataset from {data_path}...")
            dataset = self.load_data(data_path)
            return dataset
        except (FileNotFoundError, ValueError):
            try:
                dataset = load_dataset(
                    f"{self.hf_username}/{self.hf_dataset}-processed_{self.bsz}",
                    split=f"{split}[{slice}]",
                    verification_mode="no_checks",
                    keep_in_memory=False,
                    num_proc=None,
                )

                dataset = cast(Dataset, dataset)  # explicitly type it

                print(f"Loaded {split} dataset from HuggingFace Hub")

                dataset.set_format(type="numpy")

                return dataset
            except ValueError:
                print(
                    f"Building dataset from scratch... [split: {split}] | [bsz: {self.bsz}]"
                )

                dataset = load_dataset(
                    f"{self.tgt_hf_repo}",
                    split=f"{split}[{slice}]",
                    verification_mode="no_checks",
                    trust_remote_code=True,
                    keep_in_memory=False,
                    num_proc=None,
                ).select_columns(self.col_name)

                dataset = self.take_subset(split, dataset, 2_000)

                def dataset_map_fn(func: Callable) -> Dataset | DatasetDict:
                    return dataset.map(
                        func,
                        batched=True,
                        batch_size=self.bsz,
                        keep_in_memory=False,  # type: ignore
                        drop_last_batch=True,
                        num_proc=None,  # type: ignore
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

                dataset.set_format(type="numpy")

                self.save_data(split, dataset=dataset, path=data_path)

                self.upload_dataset(
                    split,
                    dataset,
                    f"{self.hf_username}/{self.hf_dataset}-processed_{self.bsz}",
                    upload_to_hub,
                )

                return dataset
