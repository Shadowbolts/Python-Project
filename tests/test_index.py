from pathlib import Path

from findex.index import DocMeta, InvertedIndex, Posting, build_index


def test_posting_hashable_and_set() -> None:
    p1 = Posting(doc_id=1, tf=2, positions=(0, 5))
    p2 = Posting(doc_id=1, tf=2, positions=(0, 5))
    p3 = Posting(doc_id=2, tf=1, positions=(3,))

    assert hash(p1) == hash(p2)
    assert p1 == p2
    s = {p1, p2, p3}
    assert len(s) == 2
    assert p1 in s
    assert p3 in s


def test_doc_meta_hashable_and_set() -> None:
    m1 = DocMeta(doc_id=0, path="doc.txt", title="Doc", length=10)
    m2 = DocMeta(doc_id=0, path="doc.txt", title="Doc", length=10)
    m3 = DocMeta(doc_id=1, path="other.txt", title="Other", length=15)

    assert hash(m1) == hash(m2)
    assert m1 == m2
    s = {m1, m2, m3}
    assert len(s) == 2


def test_build_index_basic(tmp_path: Path) -> None:
    doc1 = tmp_path / "doc1.txt"
    doc2 = tmp_path / "doc2.txt"

    doc1.write_text("Hello world hello!", encoding="utf-8")
    doc2.write_text("World of python and code.", encoding="utf-8")

    index = build_index(tmp_path)

    assert isinstance(index, InvertedIndex)
    assert index.num_documents == 2
    assert "hello" in index.postings
    assert "world" in index.postings

    hello_postings = index.postings["hello"]
    assert len(hello_postings) == 1
    assert hello_postings[0].doc_id == 0
    assert hello_postings[0].tf == 2

    world_postings = index.postings["world"]
    assert len(world_postings) == 2
    assert world_postings[0].doc_id == 0
    assert world_postings[0].tf == 1
    assert world_postings[1].doc_id == 1
    assert world_postings[1].tf == 1

    assert index.doc_lengths[0] == 3
    assert index.doc_lengths[1] == 5


def test_build_index_postings_sorted(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"doc_{i}.txt").write_text(f"common word doc_{i}", encoding="utf-8")

    index = build_index(tmp_path)
    common_postings = index.postings["common"]
    assert len(common_postings) == 5

    doc_ids = [p.doc_id for p in common_postings]
    assert doc_ids == sorted(doc_ids)
    assert len(doc_ids) == len(set(doc_ids))


def test_build_index_with_positions(tmp_path: Path) -> None:
    doc = tmp_path / "test.txt"
    doc.write_text("apple banana apple cherry apple", encoding="utf-8")

    index = build_index(tmp_path, record_positions=True)
    postings = index.postings["apple"]

    assert len(postings) == 1
    assert postings[0].tf == 3
    assert postings[0].positions == (0, 2, 4)
