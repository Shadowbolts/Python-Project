from pathlib import Path

import pytest

from findex.index import build_index, build_partial, merge


def test_build_partial_and_merge_identical(tmp_path: Path) -> None:
    doc1 = tmp_path / "a.txt"
    doc2 = tmp_path / "b.txt"
    doc3 = tmp_path / "c.txt"
    doc1.write_text("the quick brown fox jumps over the lazy dog", encoding="utf-8")
    doc2.write_text("quick brown foxes are fast and brown", encoding="utf-8")
    doc3.write_text("lazy dog sleeps under the brown tree", encoding="utf-8")

    specs = [(0, doc1), (1, doc2), (2, doc3)]
    p1 = build_partial(specs[:2], record_positions=True)
    p2 = build_partial(specs[2:], record_positions=True)
    merged_idx = merge([p1, p2])

    serial_idx = build_index(tmp_path, record_positions=True, executor="serial")
    assert merged_idx.num_docs == serial_idx.num_docs == 3
    assert merged_idx.doc_lengths == serial_idx.doc_lengths
    assert merged_idx.doc_meta == serial_idx.doc_meta
    assert merged_idx.postings == serial_idx.postings


def test_executors_produce_identical_indexes(tmp_path: Path) -> None:
    for i in range(8):
        (tmp_path / f"book_{i}.txt").write_text(
            f"chapter {i} alice wonderland sherlock code python indexing doc {i}",
            encoding="utf-8",
        )

    idx_serial = build_index(tmp_path, workers=1, executor="serial")
    idx_threads = build_index(tmp_path, workers=4, executor="threads")
    idx_procs = build_index(tmp_path, workers=4, executor="processes")

    assert idx_serial.num_docs == idx_threads.num_docs == idx_procs.num_docs == 8
    assert idx_serial.doc_lengths == idx_threads.doc_lengths == idx_procs.doc_lengths
    assert idx_serial.doc_meta == idx_threads.doc_meta == idx_procs.doc_meta
    assert idx_serial.postings == idx_threads.postings == idx_procs.postings


def test_invalid_executor_raises(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("sample content", encoding="utf-8")
    (tmp_path / "doc2.txt").write_text("sample content 2", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown executor"):
        build_index(tmp_path, workers=2, executor="invalid_executor")  # type: ignore[arg-type]
