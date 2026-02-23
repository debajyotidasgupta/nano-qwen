"""Tokenizer wrapper with support for tiktoken and sentencepiece backends."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

SPECIAL_TOKENS = {
    "bos": "<|im_start|>",
    "eos": "<|im_end|>",
    "pad": "<|im_start|>",
    "think_start": "<think>",
    "think_end": "</think>",
    "image_pad": "<|image_pad|>",
    "video_pad": "<|video_pad|>",
    "vision_start": "<|vision_start|>",
    "vision_end": "<|vision_end|>",
}


class Tokenizer:
    """Unified tokenizer interface wrapping tiktoken or sentencepiece."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = "tiktoken",
    ):
        self.backend = backend
        self._special_tokens = dict(SPECIAL_TOKENS)

        if backend == "tiktoken":
            self._init_tiktoken(model_path)
        elif backend == "sentencepiece":
            self._init_sentencepiece(model_path)
        else:
            raise ValueError(f"Unknown tokenizer backend: {backend}")

    # ----- tiktoken -----

    def _init_tiktoken(self, model_path: Optional[str]):
        import tiktoken

        if model_path and os.path.exists(model_path):
            with open(model_path, "rb") as f:
                mergeable_ranks = tiktoken.load_tiktoken_bpe(model_path)
            special = {tok: i + len(mergeable_ranks) for i, tok in enumerate(self._special_tokens.values())}
            self._enc = tiktoken.Encoding(
                name="custom",
                pat_str=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
                mergeable_ranks=mergeable_ranks,
                special_tokens=special,
            )
        else:
            self._enc = tiktoken.get_encoding("cl100k_base")

        self._vocab_size = self._enc.n_vocab

    def _init_sentencepiece(self, model_path: Optional[str]):
        import sentencepiece as spm

        if model_path is None:
            raise ValueError("sentencepiece backend requires model_path")

        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)
        self._vocab_size = self._sp.GetPieceSize()

    # ----- public API -----

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def bos_id(self) -> int:
        return self.encode(self._special_tokens["bos"], allowed_special="all")[0]

    @property
    def eos_id(self) -> int:
        return self.encode(self._special_tokens["eos"], allowed_special="all")[0]

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        allowed_special: str | set = set(),
    ) -> list[int]:
        if self.backend == "tiktoken":
            ids = self._enc.encode(text, allowed_special=allowed_special)
        else:
            ids = self._sp.Encode(text)

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        if self.backend == "tiktoken":
            return self._enc.decode(ids)
        return self._sp.Decode(ids)

    def encode_batch(self, texts: list[str], **kwargs) -> list[list[int]]:
        return [self.encode(t, **kwargs) for t in texts]

    def decode_batch(self, id_lists: list[list[int]]) -> list[str]:
        return [self.decode(ids) for ids in id_lists]
