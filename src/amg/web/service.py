"""Loads a trained run once and turns requests into short MIDI pieces.

Works with the plain one-input model from amg.train and with a
composer-conditioned model (a second, integer composer input). For the
second kind the composer names are read from config.json when present,
otherwise from the composer folders of the training pieces.
"""

from __future__ import annotations

import base64
import csv
import json
import re
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from ..generate import load_run
from ..model import next_token_probs, sample_next
from ..tokens import OnsetCounter, detect_encoding, tokens_to_elements, tokens_to_midi

MIN_NOTES, MAX_NOTES = 5, 50

# Keys a composer-conditioned run might use to store its composer list.
_COMPOSER_KEYS = ("composer_names", "composer_list", "composer_vocab", "composers_index", "composer_index")


# Folder names from piano-midi.de -> how the composers are usually written.
COMPOSER_DISPLAY = {
    "albeniz": "Albéniz", "bach": "Bach", "balakir": "Balakirev", "beeth": "Beethoven",
    "borodin": "Borodin", "brahms": "Brahms", "burgm": "Burgmüller", "chopin": "Chopin",
    "debussy": "Debussy", "granados": "Granados", "grieg": "Grieg", "haydn": "Haydn",
    "liszt": "Liszt", "mendelssohn": "Mendelssohn", "mozart": "Mozart", "muss": "Mussorgsky",
    "schubert": "Schubert", "schumann": "Schumann", "tschai": "Tchaikovsky",
}


def display_name(composer: str) -> str:
    return COMPOSER_DISPLAY.get(composer, composer.replace("_", " ").title())


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_results(run: Path, config: dict) -> dict:
    """Headline validation numbers for a run, from metrics.json or else history.csv.

    "accuracy" is next-token accuracy after a full window (for a Transformer,
    its last-position accuracy), so LSTM and Transformer runs compare fairly.
    A run stopped early with Ctrl+C has no metrics.json, so the best epoch in
    history.csv (by the monitored number) is used instead.
    """
    out: dict = {}
    metrics_path = run / "metrics.json"
    if metrics_path.exists():
        try:
            val = json.loads(metrics_path.read_text()).get("val", {})
            out = {"accuracy": _float(val.get("last_accuracy", val.get("accuracy"))),
                   "top5": _float(val.get("top5"))}
        except (ValueError, AttributeError):
            out = {}
    history = run / "history.csv"
    if not out.get("accuracy") and history.exists():
        with history.open(newline="") as f:
            rows = list(csv.DictReader(f))
        monitor = config.get("monitor", "val_accuracy")
        rows = [r for r in rows if _float(r.get(monitor)) is not None]
        if rows:
            pick = min if monitor.endswith("loss") else max
            best = pick(rows, key=lambda r: _float(r[monitor]))
            out = {"accuracy": _float(best.get("val_last_accuracy") or best.get("val_accuracy")),
                   "top5": _float(best.get("val_top5")),
                   "epoch": int(_float(best.get("epoch")) or 0) + 1}
    return {k: v for k, v in out.items() if v is not None}


class GenerationError(ValueError):
    """A request the model can't serve (bad composer, bad range, ...)."""


def _composer_names(config: dict, n_expected: int | None) -> list[str]:
    for key in _COMPOSER_KEYS:
        value = config.get(key)
        if isinstance(value, dict):  # {"bach": 0, ...}
            return [name for name, _ in sorted(value.items(), key=lambda kv: kv[1])]
        if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            return list(value)
    listed = config.get("composers")
    if isinstance(listed, list) and listed and listed != ["all"]:
        return sorted(listed)
    # Fall back to the folder names of the training pieces, e.g. "schubert/schub_d760_1.mid".
    folders = set()
    for piece in config.get("train_pieces", []) + config.get("val_pieces", []):
        parts = re.split(r"[\\/]", str(piece))
        if len(parts) >= 2:
            folders.add(parts[-2])
    names = sorted(folders)
    if n_expected and len(names) != n_expected:
        return [f"composer {i}" for i in range(n_expected)]
    return names


def _read_seeds(raw, seq_len: int) -> tuple[list[list[str]], dict[str, list[list[str]]]]:
    """Accept every seeds.json layout amg.train has written.

    - [[tok, ...], ...]                           plain run
    - [{"composer": "bach", "tokens": [...]}, ...]  composer-conditioned run
    - {"bach": [[tok, ...], ...], ...}             grouped by composer
    """
    by_composer: dict[str, list[list[str]]] = {}
    plain: list[list[str]] = []
    if isinstance(raw, dict):
        for name, group in raw.items():
            by_composer.setdefault(name, []).extend(group)
    else:
        for item in raw:
            if isinstance(item, dict):
                tokens = item.get("tokens") or item.get("seed") or []
                name = item.get("composer")
                if name is not None:
                    by_composer.setdefault(str(name), []).append(tokens)
                else:
                    plain.append(tokens)
            else:
                plain.append(item)
    by_composer = {k: [s for s in v if len(s) >= seq_len] for k, v in by_composer.items()}
    by_composer = {k: v for k, v in by_composer.items() if v}
    everything = [s for s in plain if len(s) >= seq_len] + [s for v in by_composer.values() for s in v]
    return everything, by_composer


class MusicService:
    def __init__(self, run: str | Path):
        import tensorflow as tf

        self.run = Path(run)
        self.model, self.vocab, self.config, seeds = load_run(self.run)
        self.bpm_default = 90
        self.encoding = self.config.get("encoding") or detect_encoding(self.vocab.itos)

        inputs = self.model.inputs
        self.token_input = 0
        self.composer_input = None
        if len(inputs) > 1:
            # The token input is the one whose second dimension is the sequence length.
            shapes = [tuple(i.shape) for i in inputs]
            self.token_input = max(range(len(shapes)), key=lambda i: (shapes[i][1] or 0) if len(shapes[i]) > 1 else 0)
            self.composer_input = next(i for i in range(len(inputs)) if i != self.token_input)
            self.composer_shape = shapes[self.composer_input]
        self.seq_len = int(inputs[self.token_input].shape[1])

        n_composers = self._composer_count() if self.composer_input is not None else None
        self.composers = _composer_names(self.config, n_composers) if self.composer_input is not None else []

        self.seeds, self.seeds_by_composer = _read_seeds(seeds, self.seq_len)
        if not self.seeds:
            raise SystemExit(f"No seeds of {self.seq_len}+ tokens in {self.run / 'seeds.json'}.")

        self._step = tf.function(lambda x: self.model(x, training=False), reduce_retracing=True)
        self._lock = threading.Lock()  # one generation at a time keeps TF memory predictable
        self._tf = tf
        self.generate(8, temperature=0.9, top_k=20)  # warm up so the first real request is fast

    def _composer_count(self) -> int | None:
        import keras

        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Embedding) and layer.input_dim != len(self.vocab):
                return int(layer.input_dim)
        return None

    def info(self) -> dict:
        pieces = len(self.config.get("train_pieces", [])) + len(self.config.get("val_pieces", []))
        return {
            "run": self.run.name,
            "params": int(self.model.count_params()),
            "pieces": pieces,
            "results": run_results(self.run, self.config),
            "composer_names": {c: display_name(c) for c in self.composers},
            "vocab_size": len(self.vocab),
            "seq_len": self.seq_len,
            "durations": bool(self.config.get("durations", True)),
            "encoding": self.encoding,
            "arch": self.config.get("arch", "lstm"),
            "composers": self.composers,
            "trained_on": self.config.get("composers", []),
            "min_notes": MIN_NOTES,
            "max_notes": MAX_NOTES,
        }

    def _feed(self, window: list[int], composer_id: int | None):
        tokens = self._tf.constant([window], dtype=self._tf.int32)
        if self.composer_input is None:
            return tokens
        tail = self.composer_shape[1:]
        comp = np.full((1, *[d or 1 for d in tail]), composer_id, dtype=np.int32)
        feed = [None, None]
        feed[self.token_input] = tokens
        feed[self.composer_input] = self._tf.constant(comp)
        return feed

    def generate(self, notes: int, temperature: float = 0.9, top_k: int = 20,
                 composer: str | None = None, bpm: int = 90, random_seed: int | None = None) -> dict:
        if not MIN_NOTES <= notes <= MAX_NOTES:
            raise GenerationError(f"notes must be between {MIN_NOTES} and {MAX_NOTES}.")
        composer_id = None
        pool = self.seeds
        if self.composer_input is not None:
            if not composer:
                composer = self.composers[0]
            if composer not in self.composers:
                raise GenerationError(f"Unknown composer {composer!r}.")
            composer_id = self.composers.index(composer)
            pool = [s for s in self.seeds_by_composer.get(composer, []) if len(s) >= self.seq_len] or self.seeds

        rng = np.random.default_rng(random_seed)
        seed = pool[int(rng.integers(len(pool)))]
        window = self.vocab.encode(seed)[-self.seq_len:]
        out: list[int] = []
        counter = OnsetCounter()
        with self._lock:
            # Event runs need ~3 tokens per note; the cap only stops a model that never moves on.
            for _ in range(notes * 20):
                probs = next_token_probs(self._step(self._feed(window, composer_id)))
                nxt = sample_next(probs, temperature, top_k or None, banned=[self.vocab.unk_id], rng=rng)
                out.append(nxt)
                window = window[1:] + [nxt]
                if counter.add(self.vocab.itos[nxt]) >= notes:
                    break
        tokens = self.vocab.decode(out)

        with tempfile.TemporaryDirectory() as tmp:
            path = tokens_to_midi(tokens, Path(tmp) / "piece.mid", bpm=bpm)
            midi = path.read_bytes()

        stamp = time.strftime("%Y%m%d-%H%M%S")
        who = f"_{composer}" if composer else ""
        return {
            "tokens": tokens,
            "events": tokens_to_events(tokens),
            "bpm": bpm,
            "composer": composer,
            "filename": f"amg{who}_{notes}notes_{stamp}.mid",
            "midi_base64": base64.b64encode(midi).decode("ascii"),
        }


_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


def _note_name(midi: int) -> str:
    return f"{_NAMES[midi % 12]}{midi // 12 - 1}"


def tokens_to_events(tokens: list[str]) -> list[dict]:
    """Timing, MIDI pitches and a readable label for each moment, for the browser."""
    events = []
    for start, el in tokens_to_elements(tokens):
        pitches = sorted(p.midi for p in getattr(el, "pitches", ()))
        if not pitches:  # rests are silent gaps; the piano roll shows them as space
            continue
        names = " ".join(_note_name(m) for m in pitches)
        dur = float(el.quarterLength)
        events.append({"token": f"{names} · {dur:g}", "start": float(start), "dur": dur, "pitches": pitches})
    return events
