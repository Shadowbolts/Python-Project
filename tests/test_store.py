from pathlib import Path

import pytest

from findex.index import DocMeta, InvertedIndex, Posting
from findex.store import load, load_index, save, save_index


def sample_index() -> InvertedIndex:
    postings = {
        "search": [
            Posting(doc_id=0, tf=2, positions=(1, 5)),
            Posting(doc_id=1, tf=1, positions=(3,)),
        ],
        "engine": [Posting(doc_id=1, tf=1, positions=(4,))],
    }
    doc_lengths = {0: 10, 1: 15}
    doc_meta = {
        0: DocMeta(doc_id=0, path="doc0.txt", title="Doc Zero", length=10),
        1: DocMeta(doc_id=1, path="doc1.txt", title="Doc One", length=15),
    }
    return InvertedIndex(postings=postings, doc_lengths=doc_lengths, doc_meta=doc_meta)


def test_pickle_roundtrip(tmp_path: Path) -> None:
    original = sample_index()
    out_file = tmp_path / "index.pkl"

    save_index(original, out_file)
    assert out_file.exists()

    loaded = load_index(out_file)
    assert loaded.num_documents == original.num_documents
    assert loaded.num_terms == original.num_terms
    assert loaded.postings == original.postings
    assert loaded.doc_lengths == original.doc_lengths
    assert loaded.doc_meta == original.doc_meta


def test_json_roundtrip(tmp_path: Path) -> None:
    original = sample_index()
    out_file = tmp_path / "index.json"

    save(original, out_file)
    assert out_file.exists()

    loaded = load(out_file)
    assert loaded.num_documents == original.num_documents
    assert loaded.num_terms == original.num_terms
    assert loaded.postings == original.postings
    assert loaded.doc_lengths == original.doc_lengths
    assert loaded.doc_meta == original.doc_meta


def test_unsupported_format(tmp_path: Path) -> None:
    original = sample_index()
    out_file = tmp_path / "index.unknown"

    with pytest.raises(ValueError, match="Unsupported format"):
        save_index(original, out_file, format="xml")

    with pytest.raises(ValueError, match="Unsupported format"):
        load_index(out_file, format="xml")
