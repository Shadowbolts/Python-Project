"""FastAPI application for findex search API and Jinja2/htmx web UI."""

import html
import logging
import math
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from findex.models import Index
from findex.scoring import BM25, Scorer, TfIdf
from findex.search import search
from findex.snippet import extract_snippet, snippet_for_doc
from findex.store import load_index
from findex.tokenize import tokenize
from findex.web.config import get_settings
from findex.web.deps import get_index, get_started_at
from findex.web.schemas import (
    DocumentResponse,
    HealthResponse,
    SearchResponse,
    SearchResultItem,
    StatsResponse,
)

logger = logging.getLogger("findex.web")

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _render_highlighted_snippet(doc_path: str, query_terms: list[str]) -> str:
    """Read document and return XSS-safe HTML snippet with <mark> highlights."""
    path = Path(doc_path)
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

    raw = extract_snippet(
        text,
        query_terms,
        window=80,
        highlight_prefix="\x00MARK_OPEN\x00",
        highlight_suffix="\x00MARK_CLOSE\x00",
    )
    escaped = html.escape(raw)
    return escaped.replace("\x00MARK_OPEN\x00", "<mark>").replace("\x00MARK_CLOSE\x00", "</mark>")


def _execute_search(
    idx: Index,
    q: str,
    scorer_name: Literal["bm25", "tfidf"],
    page: int,
    k: int,
    *,
    html_highlight: bool = False,
) -> tuple[SearchResponse, list[dict[str, object]]]:
    """Run ranked search, paginate results, and build snippets."""
    t0 = time.perf_counter()
    active_scorer: Scorer = BM25() if scorer_name == "bm25" else TfIdf()
    limit = max(idx.num_docs, 1)
    all_matches = search(q, idx, scorer=active_scorer, k=limit)
    total = len(all_matches)
    total_pages = math.ceil(total / k) if total > 0 else 0

    start = (page - 1) * k
    page_slice = all_matches[start : start + k]
    query_terms = list(tokenize(q))

    items: list[SearchResultItem] = []
    ui_items: list[dict[str, object]] = []
    for res in page_slice:
        plain_snip = snippet_for_doc(res.path, query_terms)
        items.append(
            SearchResultItem(
                doc_id=res.doc_id,
                title=res.title,
                score=round(res.score, 6),
                snippet=plain_snip,
            )
        )
        if html_highlight:
            html_snip = _render_highlighted_snippet(res.path, query_terms)
            ui_items.append(
                {
                    "doc_id": res.doc_id,
                    "title": res.title,
                    "score": res.score,
                    "snippet_html": html_snip,
                }
            )

    took_ms = (time.perf_counter() - t0) * 1000.0
    response = SearchResponse(
        query=q,
        scorer=scorer_name,
        page=page,
        k=k,
        total=total,
        total_pages=total_pages,
        took_ms=round(took_ms, 3),
        results=items,
    )
    return response, ui_items


def _read_doc_content(path_str: str) -> str:
    path = Path(path_str)
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load search index once on startup and clean up on shutdown."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app.state.settings = settings
    app.state.started_at = time.monotonic()

    if settings.index_path.is_file():
        t0 = time.perf_counter()
        idx = load_index(settings.index_path)
        elapsed = time.perf_counter() - t0
        app.state.index = idx
        app.state.index_size_bytes = settings.index_path.stat().st_size
        logger.info(
            "Loaded index from %s (%d docs, %d terms) in %.3fs",
            settings.index_path,
            idx.num_docs,
            len(idx),
            elapsed,
        )
    else:
        app.state.index = None
        app.state.index_size_bytes = 0
        logger.warning("Index file %s not found; /health will return 503", settings.index_path)

    try:
        yield
    finally:
        app.state.index = None
        logger.info("findex web shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    application = FastAPI(
        title="findex Search API",
        description="Full-text search engine HTTP API and Web UI.",
        version="0.7.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def request_id_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = req_id
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            logger.exception(
                "request_id=%s method=%s path=%s status=500 duration_ms=%.2f",
                req_id,
                request.method,
                request.url.path,
                duration_ms,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": req_id},
                headers={"X-Request-ID": req_id},
            )
        duration_ms = (time.perf_counter() - t0) * 1000.0
        response.headers["X-Request-ID"] = req_id
        logger.info(
            "request_id=%s method=%s path=%s status=%d duration_ms=%.2f",
            req_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    if STATIC_DIR.is_dir():
        application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request) -> Response:
        idx: Index | None = getattr(request.app.state, "index", None)
        if idx is None:
            payload = HealthResponse(status="unavailable", index_loaded=False)
            return JSONResponse(status_code=503, content=payload.model_dump())
        payload = HealthResponse(status="ok", index_loaded=True)
        return JSONResponse(status_code=200, content=payload.model_dump())

    @application.get("/stats", response_model=StatsResponse, tags=["system"])
    def stats(
        request: Request,
        index: Annotated[Index, Depends(get_index)],
        started_at: Annotated[float, Depends(get_started_at)],
    ) -> StatsResponse:
        size_bytes = int(getattr(request.app.state, "index_size_bytes", 0))
        return StatsResponse(
            num_docs=index.num_docs,
            num_terms=len(index),
            avg_doc_length=round(index.avg_doc_length, 2),
            index_size_bytes=size_bytes,
            uptime_seconds=round(time.monotonic() - started_at, 2),
        )

    @application.get("/search", response_model=SearchResponse, tags=["search"])
    def search_api(
        index: Annotated[Index, Depends(get_index)],
        q: str = Query(..., min_length=1, max_length=200, description="Search query string"),
        k: int = Query(10, ge=1, le=100, description="Results per page"),
        scorer: Literal["bm25", "tfidf"] = Query("bm25", description="Ranking algorithm"),
        page: int = Query(1, ge=1, le=1000, description="1-based page number"),
    ) -> SearchResponse:
        resp, _ = _execute_search(index, q, scorer, page, k, html_highlight=False)
        return resp

    @application.get("/docs/{doc_id}", response_model=DocumentResponse, tags=["documents"])
    def get_document_api(
        doc_id: int,
        index: Annotated[Index, Depends(get_index)],
    ) -> DocumentResponse:
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        content = _read_doc_content(meta.path)
        return DocumentResponse(
            doc_id=meta.doc_id,
            title=meta.title,
            path=meta.path,
            length=meta.length,
            content=content,
        )

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui_home(
        request: Request,
        q: str = Query("", max_length=200),
        scorer: Literal["bm25", "tfidf"] = Query("bm25"),
        page: int = Query(1, ge=1, le=1000),
        k: int = Query(10, ge=1, le=100),
    ) -> HTMLResponse:
        idx: Index | None = getattr(request.app.state, "index", None)
        clean_q = q.strip()
        if idx is None:
            return templates.TemplateResponse(
                request,
                "index.html",
                {
                    "query": clean_q,
                    "scorer": scorer,
                    "page": page,
                    "total": 0,
                    "total_pages": 0,
                    "took_ms": 0.0,
                    "results": [],
                    "num_docs": 0,
                },
            )

        if clean_q:
            resp, ui_items = _execute_search(idx, clean_q, scorer, page, k, html_highlight=True)
            ctx: dict[str, object] = {
                "query": clean_q,
                "scorer": scorer,
                "page": resp.page,
                "total": resp.total,
                "total_pages": resp.total_pages,
                "took_ms": resp.took_ms,
                "results": ui_items,
                "num_docs": idx.num_docs,
            }
        else:
            ctx = {
                "query": "",
                "scorer": scorer,
                "page": 1,
                "total": 0,
                "total_pages": 0,
                "took_ms": 0.0,
                "results": [],
                "num_docs": idx.num_docs,
            }
        return templates.TemplateResponse(request, "index.html", ctx)

    @application.get("/search-fragment", response_class=HTMLResponse, include_in_schema=False)
    def ui_search_fragment(
        request: Request,
        index: Annotated[Index, Depends(get_index)],
        q: str = Query("", max_length=200),
        scorer: Literal["bm25", "tfidf"] = Query("bm25"),
        page: int = Query(1, ge=1, le=1000),
        k: int = Query(10, ge=1, le=100),
    ) -> HTMLResponse:
        clean_q = q.strip()
        if clean_q:
            resp, ui_items = _execute_search(index, clean_q, scorer, page, k, html_highlight=True)
            ctx: dict[str, object] = {
                "query": clean_q,
                "scorer": scorer,
                "page": resp.page,
                "total": resp.total,
                "total_pages": resp.total_pages,
                "took_ms": resp.took_ms,
                "results": ui_items,
                "num_docs": index.num_docs,
            }
        else:
            ctx = {
                "query": "",
                "scorer": scorer,
                "page": 1,
                "total": 0,
                "total_pages": 0,
                "took_ms": 0.0,
                "results": [],
                "num_docs": index.num_docs,
            }
        return templates.TemplateResponse(request, "results_partial.html", ctx)

    @application.get("/doc/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    def ui_document(
        request: Request,
        doc_id: int,
        index: Annotated[Index, Depends(get_index)],
    ) -> HTMLResponse:
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        doc = DocumentResponse(
            doc_id=meta.doc_id,
            title=meta.title,
            path=meta.path,
            length=meta.length,
            content=_read_doc_content(meta.path),
        )
        return templates.TemplateResponse(request, "document.html", {"doc": doc})

    return application


app = create_app()
