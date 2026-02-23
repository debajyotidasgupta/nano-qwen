"""Data collation utilities for batching training samples."""

from __future__ import annotations

from typing import List, Dict, Optional

import torch


class TextCollator:
    """Collates text-only samples into padded batches."""

    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([s["input_ids"] for s in batch])
        labels = torch.stack([s["labels"] for s in batch])
        return {"input_ids": input_ids, "labels": labels}


class MultimodalCollator:
    """Collates multimodal samples, stacking images and text."""

    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([s["input_ids"] for s in batch])
        labels = torch.stack([s["labels"] for s in batch])
        image_token_mask = torch.stack([s["image_token_mask"] for s in batch])

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "image_token_mask": image_token_mask,
        }

        # Concatenate pixel_values from all samples (variable number of images)
        pixel_values_list = [s["pixel_values"] for s in batch if "pixel_values" in s]
        if pixel_values_list:
            result["pixel_values"] = torch.cat(pixel_values_list, dim=0)

        return result
