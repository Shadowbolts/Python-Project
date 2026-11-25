"""Tests for NumPy vectorized scoring, memmap storage, and gap-encoded postings."""

from pathlib import Path

import numpy as np
import pytest

from findex.models import Index
from findex.scoring import BM25, TfIdf, ranked_search
from findex.vector_scoring import (
    VectorIndex,
    decode_gaps,
    encode_gaps,
    load_memmap_index,
    save_memmap_index,
    vectorized_search,
)


@pytest.mark.parametrize("query", ["alice", "alice wonderland", "wonderland rabbit", "nonexistentterm"])
@pytest.mark.parametrize("scorer_name", ["bm25", "tfidf"])
def test_vectorized_search_matches_lab3(
    session_index: Index,
    query: str,
    scorer_name: str,
) -> None:
    vindex = VectorIndex.from_index(session_index)
    lab3_scorer = BM25() if scorer_name == "bm25" else TfIdf()

    expected = ranked_search(session_index, query, scorer=lab3_scorer, k=10)
    actual = vectorized_search(
        vindex,
        query,
        scorer="bm25" if scorer_name == "bm25" else "tfidf",
        k=10,
    )

    assert [r.doc_id for r in actual] == [r.doc_id for r in expected]
    for act, exp in zip(actual, expected, strict=True):
        assert act.score == pytest.approx(exp.score, rel=1e-9)
        assert act.title == exp.title


def test_memmap_roundtrip(tmp_path: Path, session_index: Index) -> None:
    vindex = VectorIndex.from_index(session_index)
    mm_dir = tmp_path / "mm_index"
    save_memmap_index(vindex, mm_dir)

    loaded = load_memmap_index(mm_dir)
    assert loaded.num_docs == vindex.num_docs
    assert loaded.avg_doc_length == pytest.approx(vindex.avg_doc_length)

    res_mem = vectorized_search(loaded, "alice", k=5)
    res_orig = vectorized_search(vindex, "alice", k=5)
    assert [r.doc_id for r in res_mem] == [r.doc_id for r in res_orig]


def test_gap_encoding_roundtrip_and_compression() -> None:
    doc_ids = np.array([1, 3, 10, 12, 100, 105, 500, 501, 2000], dtype=np.int32)
    encoded = encode_gaps(doc_ids)
    decoded = decode_gaps(encoded)
    assert np.array_equal(decoded, doc_ids)
    assert len(encoded) < doc_ids.nbytes

    empty = np.empty(0, dtype=np.int32)
    assert np.array_equal(decode_gaps(encode_gaps(empty)), empty)
