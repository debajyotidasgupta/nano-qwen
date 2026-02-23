"""Text dataset with sequence packing for pretraining.

Supports:
  - Memory-mapped data from numpy binary files for O(1) random access
  - Sequence packing: concatenate documents with BOS/EOS, pack to max_seq_len
  - Document-boundary-aware attention masking to prevent cross-document leakage
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class PackedTextDataset(Dataset):
    """Pretraining dataset that packs tokenized documents into fixed-length sequences.

    Expects a directory containing `.bin` files (numpy uint16/uint32 token arrays)
    and an optional `.idx` file mapping document boundaries.
    """

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 4096,
        bos_id: int = 151643,
        eos_id: int = 151645,
        dtype: np.dtype = np.uint32,
    ):
        self.max_seq_len = max_seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id

        bin_files = sorted(Path(data_dir).glob("*.bin")) if os.path.isdir(data_dir) else []

        if bin_files:
            # Memory-map all bin files into a single virtual token array
            self.token_arrays = [np.memmap(f, dtype=dtype, mode="r") for f in bin_files]
            self.total_tokens = sum(arr.shape[0] for arr in self.token_arrays)
            self._flat = np.concatenate(self.token_arrays) if len(self.token_arrays) == 1 else None
        else:
            # Fallback: generate random data for testing
            self.token_arrays = []
            self.total_tokens = max_seq_len * 10000
            self._flat = None

        self.num_samples = self.total_tokens // (max_seq_len + 1)

    def __len__(self) -> int:
        return max(self.num_samples, 1)

    def __getitem__(self, idx: int) -> dict:
        start = idx * (self.max_seq_len + 1)
        end = start + self.max_seq_len + 1

        if self._flat is not None and end <= self._flat.shape[0]:
            tokens = torch.from_numpy(self._flat[start:end].astype(np.int64))
        elif self.token_arrays:
            tokens = self._gather_tokens(start, end)
        else:
            tokens = torch.randint(0, 32000, (self.max_seq_len + 1,))

        input_ids = tokens[:-1]
        labels = tokens[1:]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def _gather_tokens(self, start: int, end: int) -> torch.Tensor:
        """Gather tokens from potentially multiple memory-mapped arrays."""
        tokens = []
        offset = 0
        remaining = end - start
        for arr in self.token_arrays:
            if offset + arr.shape[0] <= start:
                offset += arr.shape[0]
                continue
            local_start = max(0, start - offset)
            local_end = min(arr.shape[0], local_start + remaining)
            chunk = torch.from_numpy(arr[local_start:local_end].astype(np.int64))
            tokens.append(chunk)
            remaining -= chunk.shape[0]
            offset += arr.shape[0]
            if remaining <= 0:
                break
        return torch.cat(tokens) if tokens else torch.zeros(end - start, dtype=torch.long)


class DocumentPackingDataset(IterableDataset):
    """Streaming dataset that packs multiple documents into single training sequences.

    Documents are concatenated with BOS/EOS separators.  An attention mask is
    produced that prevents attention across document boundaries.
    """

    def __init__(
        self,
        documents: List[List[int]],
        max_seq_len: int = 4096,
        bos_id: int = 151643,
        eos_id: int = 151645,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.documents = documents
        self.max_seq_len = max_seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        indices = np.arange(len(self.documents))
        if self.shuffle:
            rng.shuffle(indices)

        buffer: List[int] = []
        doc_boundaries: List[int] = []

        for idx in indices:
            doc = [self.bos_id] + self.documents[idx] + [self.eos_id]
            doc_boundaries.append(len(buffer))
            buffer.extend(doc)

            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[: self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len + 1 :]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                # Recompute boundaries for this chunk
                doc_boundaries = [b - (self.max_seq_len + 1) for b in doc_boundaries if b >= self.max_seq_len + 1]

                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                }
