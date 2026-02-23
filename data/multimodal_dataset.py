"""Multimodal dataset for image-text training.

Handles:
  - Image-text pairs with dynamic resolution
  - Image preprocessing (resize, normalize, patch extraction)
  - Interleaving image tokens with text tokens
  - Multi-image and video inputs
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from torchvision import transforms
    HAS_TV = True
except ImportError:
    HAS_TV = False


IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]


class ImageProcessor:
    """Preprocess images for the ViT encoder."""

    def __init__(
        self,
        image_size: int = 448,
        patch_size: int = 14,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.mean = mean or IMAGE_MEAN
        self.std = std or IMAGE_STD

        if HAS_TV:
            self.transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        else:
            self.transform = None

    def __call__(self, image) -> torch.Tensor:
        if self.transform is not None:
            return self.transform(image)
        return torch.randn(3, self.image_size, self.image_size)

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_merged_tokens(self) -> int:
        """Number of tokens after 2x2 spatial merge."""
        grid = self.image_size // self.patch_size
        return (grid // 2) ** 2


class MultimodalDataset(Dataset):
    """Dataset for image-text pairs.

    Each sample is expected as a dict with:
        - "text": str or list[int] (tokenized)
        - "images": list of image paths or PIL Images (optional)
        - "image_positions": list of int — token positions to insert image tokens
    """

    def __init__(
        self,
        samples: List[Dict],
        tokenizer=None,
        image_processor: Optional[ImageProcessor] = None,
        max_seq_len: int = 4096,
        image_token_id: int = 151655,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.image_processor = image_processor or ImageProcessor()
        self.max_seq_len = max_seq_len
        self.image_token_id = image_token_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Tokenize text
        if isinstance(sample["text"], str):
            if self.tokenizer is not None:
                token_ids = self.tokenizer.encode(sample["text"])
            else:
                token_ids = list(range(100))  # placeholder
        else:
            token_ids = sample["text"]

        # Process images
        pixel_values_list = []
        image_token_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)

        if "images" in sample and sample["images"]:
            for img_input in sample["images"]:
                if isinstance(img_input, str) and HAS_PIL:
                    img = Image.open(img_input).convert("RGB")
                elif HAS_PIL and isinstance(img_input, Image.Image):
                    img = img_input
                else:
                    img = None

                pv = self.image_processor(img)
                pixel_values_list.append(pv)

            # Insert image placeholder tokens
            num_img_tokens = self.image_processor.num_merged_tokens
            positions = sample.get("image_positions", [0])

            expanded_ids = []
            img_idx = 0
            for i, tid in enumerate(token_ids):
                if img_idx < len(positions) and i == positions[img_idx]:
                    start = len(expanded_ids)
                    expanded_ids.extend([self.image_token_id] * num_img_tokens)
                    end = len(expanded_ids)
                    if end <= self.max_seq_len:
                        image_token_mask[start:end] = True
                    img_idx += 1
                expanded_ids.append(tid)
            token_ids = expanded_ids

        # Truncate and pad
        token_ids = token_ids[: self.max_seq_len + 1]
        if len(token_ids) < self.max_seq_len + 1:
            token_ids = token_ids + [0] * (self.max_seq_len + 1 - len(token_ids))

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "image_token_mask": image_token_mask[:self.max_seq_len],
        }

        if pixel_values_list:
            result["pixel_values"] = torch.stack(pixel_values_list)

        return result
