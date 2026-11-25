"""pytest-benchmark regression tests for build_index and search."""

from pathlib import Path
from typing import Any

from findex.index import build_index
from findex.models import Index
from findex.search import search
from findex.vector_scoring import VectorIndex, vectorized_search


def test_benchmark_build_index(benchmark: Any, tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    sample = "Alice was beginning to get very tired of sitting by her sister on the bank. " * 20
    for i in range(10):
        (corpus_dir / f"doc_{i}.txt").write_text(sample, encoding="utf-8")

    idx = benchmark(build_index, corpus_dir, workers=1, executor="serial")
    assert idx.num_docs == 10


def test_benchmark_search_lab3(benchmark: Any, session_index: Index) -> None:
    res = benchmark(search, "alice wonderland", session_index)
    assert len(res) >= 1


def test_benchmark_search_numpy(benchmark: Any, session_index: Index) -> None:
    vindex = VectorIndex.from_index(session_index)
    res = benchmark(vectorized_search, vindex, "alice wonderland")
    assert len(res) >= 1
