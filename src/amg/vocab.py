"""Mapping between tokens and integer ids, saved alongside each trained model."""

from __future__ import annotations

import json
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Iterable

UNK = "<UNK>"


class Vocab:
    def __init__(self, tokens: list[str]):
        if not tokens or tokens[0] != UNK:
            raise ValueError("The first vocabulary entry must be <UNK>.")
        self.itos = list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, sequences: Iterable[list[str]], min_count: int = 1) -> "Vocab":
        """Keep tokens seen at least min_count times; everything else maps to <UNK>."""
        counts = Counter(chain.from_iterable(sequences))
        kept = sorted(
            (t for t, c in counts.items() if c >= min_count),
            key=lambda t: (-counts[t], t),
        )
        return cls([UNK, *kept])

    @property
    def unk_id(self) -> int:
        return 0

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, 0) for t in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.itos[int(i)] for i in ids]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.itos, indent=0))

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text()))
