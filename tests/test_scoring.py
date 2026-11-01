from findex.models import DocMeta, Index, Posting
from findex.scoring import BM25, Scorer, SearchResult, TfIdf, ranked_search


def make_scoring_test_index() -> Index:
    postings = {
        "rare": [
            Posting(doc_id=0, tf=1),
        ],
        "common": [
            Posting(doc_id=0, tf=1),
            Posting(doc_id=1, tf=1),
            Posting(doc_id=2, tf=1),
            Posting(doc_id=3, tf=1),
        ],
        "repeated": [
            Posting(doc_id=0, tf=1),
            Posting(doc_id=1, tf=20),
        ],
        "word": [
            Posting(doc_id=0, tf=1),
            Posting(doc_id=1, tf=1),
        ],
    }
    doc_lengths = {
        0: 10,
        1: 100,
        2: 50,
        3: 50,
    }
    doc_meta = {
        0: DocMeta(doc_id=0, path="short.txt", title="Short Doc", length=10),
        1: DocMeta(doc_id=1, path="long.txt", title="Long Doc", length=100),
        2: DocMeta(doc_id=2, path="med1.txt", title="Med 1", length=50),
        3: DocMeta(doc_id=3, path="med2.txt", title="Med 2", length=50),
    }
    return Index(postings=postings, doc_lengths=doc_lengths, doc_meta=doc_meta)


def test_scorer_protocol_conformance() -> None:
    tfidf = TfIdf()
    bm25 = BM25()

    assert isinstance(tfidf, Scorer)
    assert isinstance(bm25, Scorer)


def test_search_result_order() -> None:
    r1 = SearchResult(score=1.5, doc_id=1, title="Doc 1", path="d1.txt")
    r2 = SearchResult(score=3.2, doc_id=0, title="Doc 0", path="d0.txt")

    assert r1 < r2
    assert sorted([r2, r1]) == [r1, r2]


def test_sanity_check_rare_beats_common() -> None:
    index = make_scoring_test_index()
    results = ranked_search(index, "rare common", scorer=BM25(), k=5)

    assert len(results) > 0
    top_result = results[0]
    assert top_result.doc_id == 0


def test_sanity_check_bm25_saturation() -> None:
    index = make_scoring_test_index()
    bm25 = BM25(k1=1.5, b=0.0)

    score_1 = bm25.score("repeated", Posting(doc_id=0, tf=1), index)
    score_2 = bm25.score("repeated", Posting(doc_id=0, tf=2), index)
    score_20 = bm25.score("repeated", Posting(doc_id=0, tf=20), index)
    score_21 = bm25.score("repeated", Posting(doc_id=0, tf=21), index)

    gain_1_to_2 = score_2 - score_1
    gain_20_to_21 = score_21 - score_20

    assert gain_1_to_2 > gain_20_to_21 * 5


def test_sanity_check_short_beats_long() -> None:
    index = make_scoring_test_index()
    bm25 = BM25(k1=1.5, b=0.75)

    results = ranked_search(index, "word", scorer=bm25, k=2)

    assert len(results) == 2
    assert results[0].doc_id == 0
    assert results[1].doc_id == 1
    assert results[0].score > results[1].score


def test_ranked_search_tfidf_vs_bm25() -> None:
    index = make_scoring_test_index()

    tfidf_results = ranked_search(index, "word", scorer=TfIdf(), k=2)
    bm25_results = ranked_search(index, "word", scorer=BM25(), k=2)

    assert len(tfidf_results) == 2
    assert len(bm25_results) == 2
    assert all(r.score > 0 for r in tfidf_results)
    assert all(r.score > 0 for r in bm25_results)
