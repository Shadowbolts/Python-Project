"""Pydantic boundary models for FastAPI request/response serialization."""

from typing import Literal

from pydantic import BaseModel, Field


class SearchResultItem(BaseModel):
    doc_id: int
    title: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    scorer: Literal["bm25", "tfidf"]
    page: int = Field(ge=1)
    k: int = Field(ge=1, le=100)
    total: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    took_ms: float = Field(ge=0.0)
    results: list[SearchResultItem]


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    path: str
    length: int
    content: str


class StatsResponse(BaseModel):
    num_docs: int
    num_terms: int
    avg_doc_length: float
    index_size_bytes: int
    uptime_seconds: float


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    index_loaded: bool
