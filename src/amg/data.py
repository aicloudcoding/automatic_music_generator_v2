"""Finding MIDI files, tokenizing them (with a cache), splitting and windowing."""

from __future__ import annotations

import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .tokens import TOKENIZER_VERSION, midi_to_tokens

MIDI_SUFFIXES = {".mid", ".midi"}


def find_midi_files(data_dir: str | Path, composers: list[str] | None = None) -> list[Path]:
    """All MIDI files under data_dir, optionally limited to composer subfolders.

    Extensions are matched case-insensitively, so .MID files are included.
    """
    root = Path(data_dir)
    if composers and composers != ["all"]:
        missing = [c for c in composers if not (root / c).is_dir()]
        if missing:
            available = sorted(p.name for p in root.iterdir() if p.is_dir())
            raise FileNotFoundError(f"No folder for {missing} in {root}. Available: {available}")
        dirs = [root / c for c in composers]
    else:
        dirs = [root]
    files = [p for d in dirs for p in d.rglob("*") if p.suffix.lower() in MIDI_SUFFIXES]
    return sorted(files)


def _cache_path(cache_dir: Path, midi: Path, durations: bool, encoding: str = "chords") -> Path:
    tag = f"v{TOKENIZER_VERSION}-{encoding}-{'dur' if durations else 'nodur'}"
    return cache_dir / tag / f"{midi.parent.name}__{midi.stem}.json"


def _tokenize_one(args) -> tuple[str, list[str] | None, str | None]:
    midi, durations, encoding = args
    try:
        return str(midi), midi_to_tokens(midi, durations=durations, encoding=encoding), None
    except Exception as exc:  # a few MIDI files in public collections are malformed
        return str(midi), None, f"{type(exc).__name__}: {exc}"


def load_token_sequences(
    files: list[Path],
    durations: bool = True,
    cache_dir: str | Path = "data/cache",
    workers: int | None = None,
    encoding: str = "chords",
) -> dict[str, list[str]]:
    """Tokenize each file, reusing cached results when the MIDI hasn't changed."""
    cache_dir = Path(cache_dir)
    result: dict[str, list[str]] = {}
    todo: list[Path] = []
    for f in files:
        cp = _cache_path(cache_dir, f, durations, encoding)
        if cp.exists():
            cached = json.loads(cp.read_text())
            if cached.get("mtime") == f.stat().st_mtime:
                result[str(f)] = cached["tokens"]
                continue
        todo.append(f)

    if todo:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = pool.map(_tokenize_one, [(f, durations, encoding) for f in todo])
            for path, tokens, err in tqdm(jobs, total=len(todo), desc="Parsing MIDI"):
                if err:
                    print(f"  skipped {path}: {err}")
                    continue
                f = Path(path)
                cp = _cache_path(cache_dir, f, durations, encoding)
                cp.parent.mkdir(parents=True, exist_ok=True)
                cp.write_text(json.dumps({"mtime": f.stat().st_mtime, "tokens": tokens}))
                result[path] = tokens

    # Keep the input order so splits are reproducible.
    return {str(f): result[str(f)] for f in files if str(f) in result}


def split_by_piece(
    pieces: list[str], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Put whole pieces in either train or validation, never both."""
    pieces = sorted(pieces)
    rng = random.Random(seed)
    rng.shuffle(pieces)
    n_val = max(1, round(len(pieces) * val_fraction)) if len(pieces) > 1 else 0
    return pieces[n_val:], pieces[:n_val]


def composer_of(path: str | Path) -> str:
    """The composer is the name of the folder the MIDI file sits in."""
    return Path(path).parent.name


def make_windows(
    sequences: list[list[int]],
    seq_len: int,
    stride: int = 1,
    labels: list[int] | None = None,
):
    """Sliding windows of seq_len ids as inputs; the next id as the target.

    Returns (x, y), or (x, y, c) when labels is given, where c holds each
    window's label (the composer id of the piece it came from).
    """
    xs, ys, cs = [], [], []
    for n, seq in enumerate(sequences):
        arr = np.asarray(seq, dtype=np.int32)
        if len(arr) <= seq_len:
            continue
        idx = np.arange(0, len(arr) - seq_len, stride)
        xs.append(np.stack([arr[i : i + seq_len] for i in idx]))
        ys.append(arr[idx + seq_len])
        if labels is not None:
            cs.append(np.full(len(idx), labels[n], dtype=np.int32))
    if not xs:
        x, y = np.zeros((0, seq_len), np.int32), np.zeros((0,), np.int32)
        return (x, y, np.zeros((0,), np.int32)) if labels is not None else (x, y)
    x, y = np.concatenate(xs), np.concatenate(ys)
    return (x, y, np.concatenate(cs)) if labels is not None else (x, y)


def transpose_table(itos: list[str], max_shift: int) -> np.ndarray:
    """Lookup table: row s + max_shift maps each token id to its id shifted s semitones.

    -1 marks a token that can't be shifted that far (off the keyboard or not
    in the vocabulary). Only event tokens (P60 / T0.5) are supported.
    """
    from .tokens import transpose_tokens

    stoi = {t: i for i, t in enumerate(itos)}
    table = np.full((2 * max_shift + 1, len(itos)), -1, dtype=np.int32)
    for s in range(-max_shift, max_shift + 1):
        for i, tok in enumerate(itos):
            if tok.startswith("<"):
                table[s + max_shift, i] = i
                continue
            moved = transpose_tokens([tok], s)
            if moved is not None and moved[0] in stoi:
                table[s + max_shift, i] = stoi[moved[0]]
    return table


def _base_dataset():
    import keras

    return keras.utils.PyDataset


class WindowDataset(_base_dataset()):
    """Batches of (window, next token) cut from token-id sequences on the fly.

    Nothing is copied up front, so long windows and big corpora fit in memory.
    Optional:
      labels            composer id per sequence -> inputs become {"tokens", "composer"}
      transpose + table random key shift per window each epoch (event encoding)
      windows_per_epoch  sample this many windows per epoch instead of all of them
      all_targets       target = the window shifted by one (every position), for Transformers
    """

    def __init__(self, sequences, seq_len, batch_size=256, stride=1, labels=None,
                 shuffle=True, transpose=0, table=None, windows_per_epoch=0, seed=0,
                 all_targets=False, **kwargs):
        super().__init__(**kwargs)
        self.all_targets = all_targets
        self.seq_len, self.batch_size = seq_len, batch_size
        self.labels_on = labels is not None
        self.shuffle, self.transpose, self.table = shuffle, transpose, table
        self.windows_per_epoch = windows_per_epoch
        self.rng = np.random.default_rng(seed)
        arrays = [np.asarray(s, dtype=np.int32) for s in sequences]
        self.flat = np.concatenate(arrays) if arrays else np.zeros(0, np.int32)
        starts, labs, offset = [], [], 0
        for n, arr in enumerate(arrays):
            if len(arr) > seq_len:
                idx = np.arange(0, len(arr) - seq_len, stride) + offset
                starts.append(idx)
                if labels is not None:
                    labs.append(np.full(len(idx), labels[n], dtype=np.int32))
            offset += len(arr)
        self.starts = np.concatenate(starts) if starts else np.zeros(0, np.int64)
        self.start_labels = np.concatenate(labs) if labs else np.zeros(0, np.int32)
        self._ramp = np.arange(seq_len + 1)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.order) / self.batch_size))

    def on_epoch_end(self):
        n = len(self.starts)
        if self.windows_per_epoch and self.windows_per_epoch < n:
            self.order = self.rng.choice(n, size=self.windows_per_epoch, replace=False)
        else:
            self.order = self.rng.permutation(n) if self.shuffle else np.arange(n)

    def targets(self) -> np.ndarray:
        """Next-token ids for every window, in order (unshuffled, untransposed)."""
        return self.flat[self.starts + self.seq_len]

    def __getitem__(self, i):
        pick = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        win = self.flat[self.starts[pick, None] + self._ramp]  # (batch, seq_len + 1)
        if self.transpose and self.table is not None:
            shifts = self.rng.integers(-self.transpose, self.transpose + 1, size=len(win))
            moved = self.table[shifts[:, None] + self.transpose, win]
            ok = (moved >= 0).all(axis=1)  # windows that would leave the keyboard stay put
            win = np.where(ok[:, None], moved, win)
        x, y = win[:, :-1], (win[:, 1:] if self.all_targets else win[:, -1])
        if self.labels_on:
            return {"tokens": x, "composer": self.start_labels[pick].reshape(-1, 1)}, y
        return x, y
