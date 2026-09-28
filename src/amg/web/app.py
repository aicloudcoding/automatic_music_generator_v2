"""FastAPI app: a page with note-count and style controls, plus a JSON generate endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .service import MAX_NOTES, MIN_NOTES, GenerationError, MusicService

STATIC = Path(__file__).parent / "static"


class GenerateRequest(BaseModel):
    notes: int = Field(16, ge=MIN_NOTES, le=MAX_NOTES, description="How many notes/chords to write.")
    temperature: float = Field(0.9, ge=0.0, le=2.0)
    top_k: int = Field(20, ge=0, le=200)
    composer: str | None = None
    bpm: int = Field(90, ge=30, le=240)
    random_seed: int | None = None


def create_app(service: MusicService) -> FastAPI:
    app = FastAPI(title="Automatic Music Generator", version="2.0")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def info():
        return service.info()

    @app.post("/api/generate")
    def generate(req: GenerateRequest):
        # A plain `def` runs in FastAPI's thread pool, so the page stays responsive.
        try:
            return service.generate(req.notes, req.temperature, req.top_k,
                                    req.composer, req.bpm, req.random_seed)
        except GenerationError as e:
            raise HTTPException(status_code=422, detail=str(e))

    return app
