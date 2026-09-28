"""Generate new music from a trained run.  Run: python -m amg.generate --help"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .model import next_token_probs, sample_next
from .tokens import OnsetCounter, midi_to_tokens, tokens_to_midi
from .vocab import Vocab


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MIDI from a trained model.")
    p.add_argument("--run", required=True, help="Run folder created by amg.train.")
    p.add_argument("--composer", default=None,
                   help="Style to write in (a composer the run was trained on). Default: random.")
    p.add_argument("--list-composers", action="store_true", help="Show the run's composers and exit.")
    p.add_argument("--length", type=int, default=300, help="Number of new tokens.")
    p.add_argument("--notes", type=int, default=None,
                   help="Write this many moments (a note or a chord struck together) instead of "
                        "--length tokens. Works the same for both encodings.")
    p.add_argument("--temperature", type=float, default=0.9,
                   help="Lower = safer and more repetitive; 0 = always most likely.")
    p.add_argument("--top-k", type=int, default=20, help="Sample only from the k likeliest tokens (0 = off).")
    p.add_argument("--count", type=int, default=1, help="How many pieces to generate.")
    p.add_argument("--seed-midi", default=None, help="Start from the opening of this MIDI file instead.")
    p.add_argument("--seed-index", type=int, default=None, help="Which saved seed to use (default: random).")
    p.add_argument("--include-seed", action="store_true", help="Keep the seed at the start of the output.")
    p.add_argument("--bpm", type=int, default=90)
    p.add_argument("--random-seed", type=int, default=None, help="Fix for repeatable output.")
    p.add_argument("--out-dir", default=None, help="Default: <run>/generated")
    return p.parse_args(argv)


def load_run(run: str | Path):
    import keras

    from . import layers  # noqa: F401  registers custom layers so Transformer runs load

    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    vocab = Vocab.load(run / "vocab.json")
    model_path = run / "model.keras" if (run / "model.keras").exists() else run / "best.keras"
    model = keras.models.load_model(model_path, compile=False)
    seeds = json.loads((run / "seeds.json").read_text())
    # Runs from before composer support stored seeds as plain token lists.
    seeds = [s if isinstance(s, dict) else {"composer": None, "tokens": s} for s in seeds]
    return model, vocab, config, seeds


def generate_tokens(model, vocab: Vocab, seed_tokens: list[str], length: int,
                    temperature: float = 0.9, top_k: int | None = 20,
                    rng: np.random.Generator | None = None,
                    composer_id: int | None = None, max_notes: int | None = None) -> list[str]:
    """Extend the seed one token at a time, sliding the input window forward.

    composer_id is required for models trained with the composer input.
    With max_notes, generation stops after that many moments instead of
    after `length` tokens (length then only caps runaway output).
    """
    import tensorflow as tf

    conditioned = len(model.inputs) > 1
    if conditioned and composer_id is None:
        raise ValueError("This model needs a composer id.")
    comp = tf.constant([[composer_id or 0]], dtype=tf.int32)

    # A compiled step avoids Python overhead on every one of the `length` calls.
    if conditioned:
        step = tf.function(lambda x: model({"tokens": x, "composer": comp}, training=False),
                           reduce_retracing=True)
    else:
        step = tf.function(lambda x: model(x, training=False), reduce_retracing=True)

    seq_len = model.inputs[0].shape[1]
    window = vocab.encode(seed_tokens)[-seq_len:]
    if len(window) < seq_len:
        raise ValueError(f"Seed needs at least {seq_len} tokens, got {len(window)}.")
    out: list[int] = []
    counter = OnsetCounter()
    limit = max(length, max_notes * 20) if max_notes else length
    for _ in range(limit):
        probs = next_token_probs(step(tf.constant([window], dtype=tf.int32)))
        nxt = sample_next(probs, temperature, top_k or None,
                          banned=[vocab.unk_id], rng=rng)
        out.append(nxt)
        window = window[1:] + [nxt]
        if max_notes and counter.add(vocab.itos[nxt]) >= max_notes:
            break
    return vocab.decode(out)


def main(argv=None) -> list[Path]:
    args = parse_args(argv)
    model, vocab, config, seeds = load_run(args.run)
    composers = config.get("composers", []) if config.get("use_composer") else []

    if args.list_composers:
        print("Composers:", ", ".join(composers) if composers
              else "(this run has no composer input)")
        return []
    if args.composer and not composers:
        print("Note: this run was trained without the composer input, so --composer can't change its style.")
    if args.composer and composers and args.composer not in composers:
        raise SystemExit(f"Unknown composer {args.composer!r}. Choose from: {', '.join(composers)}")

    rng = np.random.default_rng(args.random_seed)
    seq_len = model.inputs[0].shape[1]

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.run) / "generated"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    written = []
    for n in range(args.count):
        composer = args.composer or (str(rng.choice(composers)) if composers else None)

        if args.seed_midi:
            pool = [midi_to_tokens(args.seed_midi, durations=config["durations"],
                                   encoding=config.get("encoding", "chords"))[:seq_len]]
        else:
            # Prefer seeds from the chosen composer; fall back to any seed.
            pool = [s["tokens"] for s in seeds if composer and s["composer"] == composer]
            pool = pool or [s["tokens"] for s in seeds]
        if args.seed_index is not None:
            seed = pool[args.seed_index % len(pool)]
        else:
            seed = pool[int(rng.integers(len(pool)))]

        cid = composers.index(composer) if composers else None
        new = generate_tokens(model, vocab, seed, args.length, args.temperature, args.top_k,
                              rng, composer_id=cid, max_notes=args.notes)
        tokens = list(seed) + new if args.include_seed else new
        tag = f"{composer}_" if composer else ""
        name = f"gen_{tag}{stamp}_{n + 1}_t{args.temperature}_k{args.top_k}.mid"
        path = tokens_to_midi(tokens, out_dir / name, bpm=args.bpm)
        written.append(path)
        print("Wrote", path)
    return written


if __name__ == "__main__":
    main()
