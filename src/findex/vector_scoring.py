"""
Vectorized NumPy scoring (BM25 / TF-IDF), memmap index storage, and gap-encoded postings.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from findex.models import DocMeta, Index
from findex.scoring import SearchResult
from findex.tokenize import tokenize


def encode_gaps(doc_ids: NDArray[np.int32]) -> bytes:
    """Delta-encode sorted int32 doc_ids and pack with unsigned LEB128 varints."""
    if doc_ids.size == 0:
        return b""
    gaps = np.empty_like(doc_ids)
    gaps[0] = doc_ids[0]
    if doc_ids.size > 1:
        gaps[1:] = np.diff(doc_ids)

    out = bytearray()
    for val in gaps.tolist():
        v = int(val)
        while v >= 0x80:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        out.append(v)
    return bytes(out)


def decode_gaps(data: bytes) -> NDArray[np.int32]:
    """Decode unsigned LEB128 varint gaps and reconstruct doc_ids via np.cumsum."""
    if not data:
        return np.empty(0, dtype=np.int32)
    gaps: list[int] = []
    val = 0
    shift = 0
    for byte in data:
        val |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
        else:
            gaps.append(val)
            val = 0
            shift = 0
    arr = np.asarray(gaps, dtype=np.int32)
    return np.cumsum(arr, dtype=np.int32)


@dataclass
class VectorIndex:
    """Contiguous NumPy int32 representation of an inverted index (in-memory or memmap)."""

    num_docs: int
    avg_doc_length: float
    doc_len: NDArray[np.int32]
    doc_meta: dict[int, DocMeta]
    term_slices: dict[str, tuple[int, int]]
    doc_ids_buf: NDArray[np.int32]
    tfs_buf: NDArray[np.int32]

    @classmethod
    def from_index(cls, index: Index) -> VectorIndex:
        """Convert an Index into contiguous int32 NumPy arrays."""
        max_doc_id = max(index.doc_meta.keys(), default=-1)
        size = max(index.num_docs, max_doc_id + 1)
        doc_len = np.zeros(size, dtype=np.int32)
        for doc_id, length in index.doc_lengths.items():
            doc_len[doc_id] = length

        total_postings = sum(len(plist) for plist in index.postings.values())
        doc_ids_buf = np.empty(total_postings, dtype=np.int32)
        tfs_buf = np.empty(total_postings, dtype=np.int32)
        term_slices: dict[str, tuple[int, int]] = {}

        offset = 0
        for term, plist in index.postings.items():
            count = len(plist)
            end = offset + count
            if count > 0:
                doc_ids_buf[offset:end] = [p.doc_id for p in plist]
                tfs_buf[offset:end] = [p.tf for p in plist]
            term_slices[term] = (offset, end)
            offset = end

        return cls(
            num_docs=index.num_docs,
            avg_doc_length=index.avg_doc_length,
            doc_len=doc_len,
            doc_meta=index.doc_meta,
            term_slices=term_slices,
            doc_ids_buf=doc_ids_buf,
            tfs_buf=tfs_buf,
        )

    def term_arrays(self, term: str) -> tuple[NDArray[np.int32], NDArray[np.int32]] | None:
        bounds = self.term_slices.get(term)
        if bounds is None:
            return None
        start, end = bounds
        return self.doc_ids_buf[start:end], self.tfs_buf[start:end]


def save_memmap_index(vindex: VectorIndex, directory: Path) -> None:
    """Persist VectorIndex to disk in raw binary + .npy files for zero-copy memmap loading."""
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "doc_len.npy", vindex.doc_len)
    vindex.doc_ids_buf.tofile(directory / "doc_ids.bin")
    vindex.tfs_buf.tofile(directory / "tfs.bin")

    meta_payload: dict[str, Any] = {
        "num_docs": vindex.num_docs,
        "avg_doc_length": vindex.avg_doc_length,
        "total_postings": int(vindex.doc_ids_buf.size),
        "term_slices": {t: [s, e] for t, (s, e) in vindex.term_slices.items()},
        "doc_meta": {
            str(k): {"path": m.path, "title": m.title, "length": m.length}
            for k, m in vindex.doc_meta.items()
        },
    }
    (directory / "meta.json").write_text(json.dumps(meta_payload), encoding="utf-8")


def load_memmap_index(directory: Path) -> VectorIndex:
    """Load VectorIndex using read-only np.memmap for postings buffers."""
    raw = cast(dict[str, Any], json.loads((directory / "meta.json").read_text(encoding="utf-8")))
    doc_len = cast(NDArray[np.int32], np.load(directory / "doc_len.npy", mmap_mode="r"))

    total_postings = int(raw["total_postings"])
    if total_postings > 0:
        doc_ids_buf = cast(
            NDArray[np.int32],
            np.memmap(directory / "doc_ids.bin", dtype=np.int32, mode="r", shape=(total_postings,)),
        )
        tfs_buf = cast(
            NDArray[np.int32],
            np.memmap(directory / "tfs.bin", dtype=np.int32, mode="r", shape=(total_postings,)),
        )
    else:
        doc_ids_buf = np.empty(0, dtype=np.int32)
        tfs_buf = np.empty(0, dtype=np.int32)

    raw_slices = cast(dict[str, list[int]], raw["term_slices"])
    term_slices = {t: (int(v[0]), int(v[1])) for t, v in raw_slices.items()}

    raw_meta = cast(dict[str, dict[str, Any]], raw["doc_meta"])
    doc_meta = {
        int(k): DocMeta(doc_id=int(k), path=str(v["path"]), title=str(v["title"]), length=int(v["length"]))
        for k, v in raw_meta.items()
    }

    return VectorIndex(
        num_docs=int(raw["num_docs"]),
        avg_doc_length=float(raw["avg_doc_length"]),
        doc_len=doc_len,
        doc_meta=doc_meta,
        term_slices=term_slices,
        doc_ids_buf=doc_ids_buf,
        tfs_buf=tfs_buf,
    )


def vectorized_search(
    vindex: VectorIndex,
    query: str,
    *,
    scorer: Literal["bm25", "tfidf"] = "bm25",
    k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[SearchResult]:
    """Rank documents using vectorized NumPy ufuncs and np.argpartition top-k."""
    tokens = list(tokenize(query))
    if not tokens or vindex.num_docs == 0 or k <= 0:
        return []

    scores = np.zeros(vindex.doc_len.shape[0], dtype=np.float64)
    n_docs = vindex.num_docs
    avg_len = vindex.avg_doc_length or 1.0
    any_matched = False

    for token in tokens:
        arrays = vindex.term_arrays(token)
        if arrays is None:
            continue
        doc_ids, tfs = arrays
        df = int(doc_ids.size)
        if df == 0:
            continue
        any_matched = True

        if scorer == "bm25":
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            len_norm = 1.0 - b + b * (vindex.doc_len[doc_ids] / avg_len)
            term_scores = idf * (tfs * (k1 + 1.0)) / (tfs + k1 * len_norm)
        else:
            idf = math.log((n_docs + 1.0) / (df + 1.0)) + 1.0
            term_scores = (1.0 + np.log(tfs)) * idf

        scores[doc_ids] += term_scores

    if not any_matched:
        return []

    nonzero_ids = np.flatnonzero(scores > 0.0)
    if nonzero_ids.size == 0:
        return []

    if nonzero_ids.size > k:
        cand_scores = scores[nonzero_ids]
        top_local = np.argpartition(cand_scores, -k)[-k:]
        top_ids = nonzero_ids[top_local]
    else:
        top_ids = nonzero_ids

    # Sort the k candidates descending by score (then doc_id for deterministic ties)
    order = sorted(
        (int(did) for did in top_ids.tolist()),
        key=lambda did: (float(scores[did]), did),
        reverse=True,
    )

    results: list[SearchResult] = []
    for doc_id in order:
        meta = vindex.doc_meta[doc_id]
        results.append(
            SearchResult(
                score=float(scores[doc_id]),
                doc_id=doc_id,
                title=meta.title,
                path=meta.path,
            )
        )
    return results
