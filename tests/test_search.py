from findex.index import DocMeta, InvertedIndex, Posting
from findex.search import (
    execute_boolean_query,
    merge_and,
    merge_diff,
    merge_or,
    set_and,
    set_diff,
    set_or,
)


def test_merge_primitives() -> None:
    a = [1, 3, 5, 7, 9]
    b = [2, 3, 6, 7, 10]

    assert merge_and(a, b) == [3, 7]
    assert merge_or(a, b) == [1, 2, 3, 5, 6, 7, 9, 10]
    assert merge_diff(a, b) == [1, 5, 9]

    assert merge_and(a, b) == set_and(a, b)
    assert merge_or(a, b) == set_or(a, b)
    assert merge_diff(a, b) == set_diff(a, b)


def test_merge_disjoint_and_identical() -> None:
    a = [1, 2, 3]
    b = [4, 5, 6]
    assert merge_and(a, b) == []
    assert merge_or(a, b) == [1, 2, 3, 4, 5, 6]
    assert merge_diff(a, b) == [1, 2, 3]

    assert merge_and(a, a) == [1, 2, 3]
    assert merge_or(a, a) == [1, 2, 3]
    assert merge_diff(a, a) == []


def make_test_index() -> InvertedIndex:
    postings = {
        "apple": [Posting(doc_id=0, tf=1), Posting(doc_id=1, tf=2)],
        "banana": [Posting(doc_id=1, tf=1), Posting(doc_id=2, tf=3)],
        "cherry": [Posting(doc_id=0, tf=2), Posting(doc_id=2, tf=1)],
    }
    doc_lengths = {0: 3, 1: 3, 2: 4}
    doc_meta = {
        0: DocMeta(doc_id=0, path="d0.txt", title="Doc 0", length=3),
        1: DocMeta(doc_id=1, path="d1.txt", title="Doc 1", length=3),
        2: DocMeta(doc_id=2, path="d2.txt", title="Doc 2", length=4),
    }
    return InvertedIndex(postings=postings, doc_lengths=doc_lengths, doc_meta=doc_meta)


def test_boolean_search_implicit_and() -> None:
    index = make_test_index()
    res_merge = execute_boolean_query(index, "apple banana", engine="merge")
    res_set = execute_boolean_query(index, "apple banana", engine="set")
    assert res_merge == [1]
    assert res_set == [1]


def test_boolean_search_or() -> None:
    index = make_test_index()
    res_merge = execute_boolean_query(index, "apple OR cherry", engine="merge")
    res_set = execute_boolean_query(index, "apple OR cherry", engine="set")
    assert res_merge == [0, 1, 2]
    assert res_set == [0, 1, 2]


def test_boolean_search_not() -> None:
    index = make_test_index()
    res_merge = execute_boolean_query(index, "apple NOT banana", engine="merge")
    res_set = execute_boolean_query(index, "apple NOT banana", engine="set")
    assert res_merge == [0]
    assert res_set == [0]


def test_boolean_search_unknown_term() -> None:
    index = make_test_index()
    res_merge = execute_boolean_query(index, "apple unknown", engine="merge")
    assert res_merge == []

    res_or = execute_boolean_query(index, "apple OR unknown", engine="merge")
    assert res_or == [0, 1]


def test_boolean_search_leading_not() -> None:
    index = make_test_index()
    res_merge = execute_boolean_query(index, "NOT banana", engine="merge")
    res_set = execute_boolean_query(index, "NOT banana", engine="set")
    assert res_merge == [0]
    assert res_set == [0]
