"""FastAPI dependency providers for index, settings, and runtime state."""

import time

from fastapi import HTTPException, Request

from findex.models import Index


def get_index(request: Request) -> Index:
    """Return the loaded search index or raise 503 if unavailable."""
    idx: Index | None = getattr(request.app.state, "index", None)
    if idx is None:
        raise HTTPException(status_code=503, detail="Index is not loaded")
    return idx


def get_started_at(request: Request) -> float:
    """Return monotonic timestamp when the app started."""
    return float(getattr(request.app.state, "started_at", time.monotonic()))
