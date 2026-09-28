"""Train the model.  Run: python -m amg.train --help"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .data import (
    WindowDataset,
    composer_of,
    find_midi_files,
    load_token_sequences,
    split_by_piece,
    transpose_table,
)
from .tokens import DURATIONS, ENCODINGS, PIANO_HIGH, PIANO_LOW, PITCH, SHIFT, transpose_tokens
from .vocab import UNK, Vocab

# Default window length per architecture and encoding. Events use ~3 tokens per
# moment: 192 tokens is ~60 moments, 512 is ~165 (Transformers handle long windows well).
DEFAULT_SEQ_LEN = {("lstm", "events"): 192, ("lstm", "chords"): 50,
                   ("transformer", "events"): 512, ("transformer", "chords"): 128}
# Per-architecture defaults for options left unset on the command line.
ARCH_DEFAULTS = {
    "lstm": {"layers": 2, "dropout": 0.3, "lr": 5e-4, "batch_size": 256, "windows_per_epoch": 0},
    # A Transformer learns from every position of a window, so a random sample of
    # windows per epoch already covers the data many times over.
    "transformer": {"layers": 6, "dropout": 0.2, "lr": 3e-4, "batch_size": 64, "windows_per_epoch": 20000},
}


def event_vocab() -> Vocab:
    """Every piano key and every time step, so any key shift stays in the vocabulary."""
    return Vocab([UNK, *(f"{PITCH}{m}" for m in range(PIANO_LOW, PIANO_HIGH + 1)),
                  *(f"{SHIFT}{d}" for d in DURATIONS)])


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the music generator.")
    p.add_argument("--data-dir", default="data/midi", help="Folder of composer subfolders.")
    p.add_argument("--composers", nargs="+", default=["schubert"],
                   help='Composer folders to train on, or "all".')
    p.add_argument("--encoding", choices=ENCODINGS, default="events",
                   help="events: one token per note with its exact pitch, plus time steps (keeps chord "
                        "voicing and octaves). chords: the original one-token-per-moment encoding.")
    p.add_argument("--no-durations", action="store_true",
                   help="Pitch-only tokens, like v1 (no rhythm). Chords encoding only.")
    p.add_argument("--seq-len", type=int, default=None,
                   help="Tokens the model sees per prediction (default: 192 events / 50 chords).")
    p.add_argument("--stride", type=int, default=1, help="Step between training windows.")
    p.add_argument("--transpose", type=int, default=0,
                   help="Also train on each piece shifted up to this many semitones up and down "
                        "(e.g. 5 = 11 keys). Training pieces only. With events each window gets a "
                        "random key every epoch, so this costs no extra memory or time.")
    p.add_argument("--no-composer", action="store_true",
                   help="Don't give the model the composer as an input.")
    p.add_argument("--min-count", type=int, default=10,
                   help="Tokens rarer than this become <UNK>.")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--arch", choices=["lstm", "transformer"], default="lstm",
                   help="lstm: stacked LSTMs. transformer: GPT-style causal self-attention, "
                        "better at long-range structure (see README).")
    p.add_argument("--embed-dim", type=int, default=128, help="LSTM: token embedding size.")
    p.add_argument("--units", type=int, default=256, help="LSTM: units per layer.")
    p.add_argument("--d-model", type=int, default=256, help="Transformer: width of every layer.")
    p.add_argument("--heads", type=int, default=4, help="Transformer: attention heads per layer.")
    p.add_argument("--layers", type=int, default=None, help="Default: 2 LSTM / 6 Transformer.")
    p.add_argument("--dropout", type=float, default=None, help="Default: 0.3 LSTM / 0.2 Transformer.")
    p.add_argument("--lr", type=float, default=None, help="Default: 5e-4 LSTM / 3e-4 Transformer.")
    p.add_argument("--clipnorm", type=float, default=1.0,
                   help="Gradient clipping: cap on the size of each update (0 = off). "
                        "Keeps LSTM training stable, especially with --mixed-precision.")
    p.add_argument("--batch-size", type=int, default=None, help="Default: 256 LSTM / 64 Transformer.")
    p.add_argument("--epochs", type=int, default=100, help="Upper limit; early stopping usually ends sooner.")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--monitor", default="val_accuracy", choices=["val_accuracy", "val_loss"],
                   help="Which validation number picks the saved weights and ends training.")
    p.add_argument("--windows-per-epoch", type=int, default=None,
                   help="Train on this many random windows per epoch (0 = all). Shorter epochs "
                        "give more frequent checkpoints. Default: all for LSTM, 20000 for Transformer.")
    p.add_argument("--mixed-precision", action="store_true", help="Faster on RTX GPUs.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--name", default=None, help="Run folder name (default: timestamp).")
    p.add_argument("--cache-dir", default="data/cache")
    args = p.parse_args(argv)
    for key, value in ARCH_DEFAULTS[args.arch].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def main(argv=None) -> Path:
    args = parse_args(argv)

    import keras
    from .check_gpu import gpu_summary
    from .layers import LastTokenAccuracy
    from .model import build_model, build_transformer

    keras.utils.set_random_seed(args.seed)
    random.seed(args.seed)
    gpus = gpu_summary()
    print("Device:", ", ".join(gpus) if gpus else "CPU only (run `python -m amg.check_gpu`)")
    if args.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    durations = not args.no_durations
    events = args.encoding == "events"
    if events and not durations:
        raise SystemExit("--no-durations only works with --encoding chords.")
    transformer = args.arch == "transformer"
    seq_len = args.seq_len or DEFAULT_SEQ_LEN[(args.arch, args.encoding)]
    files = find_midi_files(args.data_dir, args.composers)
    print(f"{len(files)} MIDI files from {args.composers} | {args.arch}, encoding: {args.encoding}, "
          f"window {seq_len}")
    seqs = load_token_sequences(files, durations=durations, cache_dir=args.cache_dir,
                                encoding=args.encoding)

    train_keys, val_keys = split_by_piece(list(seqs), args.val_fraction, args.seed)
    if not val_keys:
        raise SystemExit("Need at least 2 pieces to have a validation set.")
    print(f"Split by piece: {len(train_keys)} train / {len(val_keys)} validation")

    composers = sorted({composer_of(k) for k in seqs})
    use_composer = not args.no_composer and len(composers) > 1
    comp_id = {c: i for i, c in enumerate(composers)}
    if use_composer:
        print(f"Composer input on: {len(composers)} composers")

    table = None
    if events:
        # Fixed vocabulary; transposition happens per batch, so pieces are stored once.
        vocab = event_vocab()
        train_seqs = [seqs[k] for k in train_keys]
        train_labels = [comp_id[composer_of(k)] for k in train_keys]
        if args.transpose:
            table = transpose_table(vocab.itos, args.transpose)
            print(f"Transposition ±{args.transpose}: a random key per window, every epoch")
    else:
        # Chords encoding: add shifted copies of each training piece (validation stays in its real key).
        shifts = [s for s in range(-args.transpose, args.transpose + 1) if s != 0]
        train_seqs, train_labels, skipped = [], [], 0
        for k in train_keys:
            for shift in [0, *shifts]:
                t = transpose_tokens(seqs[k], shift)
                if t is None:  # would go off the keyboard
                    skipped += 1
                    continue
                train_seqs.append(t)
                train_labels.append(comp_id[composer_of(k)])
        if shifts:
            print(f"Transposition ±{args.transpose}: {len(train_seqs)} training sequences "
                  f"from {len(train_keys)} pieces ({skipped} shifts skipped for range)")
        # The vocabulary comes from training pieces only, so validation stays unseen.
        vocab = Vocab.build(train_seqs, min_count=args.min_count)

    train_ids = [vocab.encode(t) for t in train_seqs]
    val_ids = [vocab.encode(seqs[k]) for k in val_keys]
    val_labels = [comp_id[composer_of(k)] for k in val_keys]
    labels = (lambda ls: ls) if use_composer else (lambda ls: None)
    train_ds = WindowDataset(train_ids, seq_len, args.batch_size, args.stride, labels(train_labels),
                             shuffle=True, transpose=args.transpose if events else 0, table=table,
                             windows_per_epoch=args.windows_per_epoch, seed=args.seed,
                             all_targets=transformer)
    # Transformer windows overlap heavily at stride 1 and each scores every position,
    # so validation uses a coarser stride (still thousands of full windows).
    val_stride = max(1, seq_len // 32) if transformer else 1
    val_ds = WindowDataset(val_ids, seq_len, args.batch_size, val_stride, labels(val_labels),
                           shuffle=False, all_targets=transformer)
    if len(train_ds.starts) == 0 or len(val_ds.starts) == 0:
        raise SystemExit("Not enough tokens for the chosen --seq-len.")

    unk_rate = float(np.mean(np.concatenate([np.array(s) for s in val_ids]) == vocab.unk_id))
    print(f"Vocabulary: {len(vocab)} tokens | windows: {len(train_ds.starts):,} train, "
          f"{len(val_ds.starts):,} val | <UNK> in val: {unk_rate:.1%}")

    run_dir = Path(args.runs_dir) / (args.name or time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    vocab.save(run_dir / "vocab.json")
    config = {**vars(args), "seq_len": seq_len, "durations": durations, "vocab_size": len(vocab),
              "composers": composers, "use_composer": use_composer,
              "train_pieces": train_keys, "val_pieces": val_keys}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Seeds for generation: up to 10 starting windows per composer, from validation
    # pieces when the composer has one, otherwise from its training pieces.
    rng = np.random.default_rng(args.seed)
    seeds = []
    for c in composers:
        pool = [ids for ids, k in zip(val_ids, val_keys) if composer_of(k) == c] or \
               [vocab.encode(seqs[k]) for k in train_keys if composer_of(k) == c]
        pool = [ids for ids in pool if len(ids) > seq_len]
        for _ in range(min(10, sum(len(ids) - seq_len for ids in pool)) if pool else 0):
            ids = pool[int(rng.integers(len(pool)))]
            start = int(rng.integers(len(ids) - seq_len))
            if events:  # end on a time step so generation starts a fresh moment
                while start > 0 and not vocab.itos[ids[start + seq_len - 1]].startswith(SHIFT):
                    start -= 1
            seeds.append({"composer": c, "tokens": vocab.decode(ids[start:start + seq_len])})
    (run_dir / "seeds.json").write_text(json.dumps(seeds))

    n_comp = len(composers) if use_composer else 0
    if transformer:
        model = build_transformer(len(vocab), seq_len, args.d_model, args.layers, args.heads,
                                  dropout=args.dropout, num_composers=n_comp)
    else:
        model = build_model(len(vocab), seq_len, args.embed_dim, args.units, args.layers,
                            args.dropout, num_composers=n_comp)
    metrics = ["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")]
    if transformer:
        metrics.append(LastTokenAccuracy())  # comparable with the LSTM's accuracy
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr, clipnorm=args.clipnorm or None),
        loss="sparse_categorical_crossentropy",
        metrics=metrics,
    )
    model.summary()

    callbacks = [
        # Stop at once if the loss becomes NaN instead of training a broken model for hours.
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(monitor=args.monitor, patience=args.patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=max(2, args.patience // 3), verbose=1),
        keras.callbacks.ModelCheckpoint(str(run_dir / "best.keras"), monitor=args.monitor,
                                        save_best_only=True),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
    ]
    start = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)
    minutes = (time.time() - start) / 60
    if model.stop_training and not np.isfinite(model.history.history.get("loss", [np.nan])[-1]):
        raise SystemExit(
            "\nTraining stopped: the loss became NaN (the numbers overflowed).\n"
            "Try without --mixed-precision, a lower --lr (e.g. 2e-4) or a lower --clipnorm (e.g. 0.5)."
        )

    model.save(run_dir / "model.keras")
    # Training accuracy on a sample of windows in their real keys.
    train_eval_ds = WindowDataset(train_ids, seq_len, args.batch_size, args.stride, labels(train_labels),
                                  windows_per_epoch=20_000 if transformer else 100_000, seed=args.seed,
                                  all_targets=transformer)
    train_eval = model.evaluate(train_eval_ds, verbose=0, return_dict=True)
    val_eval = model.evaluate(val_ds, verbose=0, return_dict=True)
    y_train, y_val = train_ds.targets(), val_ds.targets()
    most_common = Counter(y_train.tolist()).most_common(1)[0][0]
    metrics = {
        "train": train_eval,
        "val": val_eval,
        "baseline_most_common_token_val_acc": float(np.mean(y_val == most_common)),
        "vocab_size": len(vocab),
        "training_minutes": round(minutes, 1),
        "device": gpus or ["CPU"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nDone in {minutes:.1f} min. Saved to {run_dir}")
    print(f"  train acc {train_eval['accuracy']:.1%} | val acc {val_eval['accuracy']:.1%} "
          f"| val top-5 {val_eval['top5']:.1%} | always-most-common baseline "
          f"{metrics['baseline_most_common_token_val_acc']:.1%}")
    if transformer:
        print(f"  val accuracy at the last position (compare with an LSTM's val acc): "
              f"{val_eval['last_accuracy']:.1%}")
    hint = f" --composer {composers[0]}" if use_composer else ""
    print(f"Try it in the browser:  python -m amg.web --run {run_dir}")
    print(f"Generate with: python -m amg.generate --run {run_dir}{hint}")
    return run_dir


if __name__ == "__main__":
    main()
