"""Run the web app:  python -m amg.web  (or --run runs/<timestamp> for a model you trained)"""

from __future__ import annotations

import argparse
from pathlib import Path


PRETRAINED = Path("pretrained/transformer")


def latest_run(runs_dir: str = "runs") -> Path | None:
    """The newest run you trained, or else the pretrained model that ships with the repo."""
    runs = sorted(p for p in Path(runs_dir).glob("*") if (p / "vocab.json").exists()
                  and any((p / f).exists() for f in ("model.keras", "best.keras")))
    if runs:
        return runs[-1]
    return PRETRAINED if (PRETRAINED / "vocab.json").exists() else None


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Serve the music generator in your browser.")
    p.add_argument("--run", default=None, help="Run folder from amg.train (default: newest in runs/, else pretrained/transformer).")
    p.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to allow other devices on your network.")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    run = Path(args.run) if args.run else latest_run()
    if run is None or not (run / "vocab.json").exists():
        raise SystemExit("No model found. Run this from the project folder, or pass --run runs/<timestamp>.")

    import uvicorn

    from .app import create_app
    from .service import MusicService

    print(f"Loading model from {run} ...")
    app = create_app(MusicService(run))
    print(f"Open http://localhost:{args.port} in your browser.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
