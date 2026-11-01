from findex.models import DocMeta, Index, Posting
from findex.search import _cached_search_doc_ids, search
from findex.snippet import extract_snippet
from findex.timing import timed


def test_timed_decorator(caplog) -> None:
    @timed
    def sample_func(x: int) -> int:
        return x * 2

    assert sample_func.__name__ == "sample_func"
    with caplog.at_level("INFO"):
        result = sample_func(5)
    assert result == 10
    assert any("sample_func took" in record.message for record in caplog.records)


def test_lru_cache_query() -> None:
    index = Index(
        postings={"python": [Posting(doc_id=0, tf=1)]},
        doc_lengths={0: 10},
        doc_meta={0: DocMeta(doc_id=0, path="d.txt", title="D", length=10)},
    )
    _cached_search_doc_ids.cache_clear()
    info_before = _cached_search_doc_ids.cache_info()
    assert info_before.hits == 0

    _ = _cached_search_doc_ids("python", index)
    info_1 = _cached_search_doc_ids.cache_info()
    assert info_1.hits == 0
    assert info_1.misses == 1

    _ = _cached_search_doc_ids("python", index)
    info_2 = _cached_search_doc_ids.cache_info()
    assert info_2.hits == 1


def test_extract_snippet_highlight() -> None:
    text = "The quick brown fox jumps over the lazy dog in the forest."
    snippet = extract_snippet(text, ["fox"], window=15)
    assert "**fox**" in snippet

    snippet_multi = extract_snippet(text, ["fox", "dog"], window=40)
    assert "**fox**" in snippet_multi
    assert "**dog**" in snippet_multi


def test_search_end_to_end() -> None:
    index = Index(
        postings={
            "alice": [Posting(doc_id=0, tf=5)],
            "wonderland": [Posting(doc_id=0, tf=3)],
        },
        doc_lengths={0: 50},
        doc_meta={0: DocMeta(doc_id=0, path="alice.txt", title="Alice Book", length=50)},
    )
    results = search("alice AND wonderland", index)
    assert len(results) == 1
    assert results[0].doc_id == 0
    assert results[0].score > 0
