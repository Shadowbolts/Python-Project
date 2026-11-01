from findex.models import DocMeta, Index, Posting
from findex.query import And, Not, Or, Phrase, Term, parse_query


def test_tree_construction_with_operators() -> None:
    t_python = Term("python")
    t_async = Term("async")
    t_await = Term("await")
    t_java = Term("java")

    tree = t_python & (t_async | t_await) & ~t_java

    expected = And(And(t_python, Or(t_async, t_await)), Not(t_java))
    assert tree == expected


def test_parser_implicit_and() -> None:
    parsed = parse_query("python async")
    expected = And(Term("python"), Term("async"))
    assert parsed == expected


def test_parser_explicit_and_or() -> None:
    parsed = parse_query("python AND async OR await")
    expected = Or(And(Term("python"), Term("async")), Term("await"))
    assert parsed == expected


def test_parser_precedence_and_over_or() -> None:
    parsed = parse_query("a OR b c")
    expected = Or(Term("a"), And(Term("b"), Term("c")))
    assert parsed == expected


def test_parser_parentheses() -> None:
    parsed = parse_query("(a OR b) AND c")
    expected = And(Or(Term("a"), Term("b")), Term("c"))
    assert parsed == expected


def test_parser_not() -> None:
    parsed = parse_query("python NOT java")
    expected = And(Term("python"), Not(Term("java")))
    assert parsed == expected


def test_parser_phrase() -> None:
    parsed = parse_query('"event loop"')
    expected = Phrase(("event", "loop"))
    assert parsed == expected


def test_parser_complex_query() -> None:
    parsed = parse_query('python AND (async OR await) NOT java "event loop"')
    expected = And(
        And(
            And(Term("python"), Or(Term("async"), Term("await"))),
            Not(Term("java")),
        ),
        Phrase(("event", "loop")),
    )
    assert parsed == expected


def test_phrase_evaluation() -> None:
    postings = {
        "event": [
            Posting(doc_id=0, tf=2, positions=(10, 40)),
            Posting(doc_id=1, tf=1, positions=(5,)),
        ],
        "loop": [
            Posting(doc_id=0, tf=2, positions=(11, 80)),
            Posting(doc_id=1, tf=1, positions=(20,)),
        ],
    }
    doc_lengths = {0: 100, 1: 100}
    doc_meta = {
        0: DocMeta(doc_id=0, path="d0.txt", title="Doc 0", length=100),
        1: DocMeta(doc_id=1, path="d1.txt", title="Doc 1", length=100),
    }
    index = Index(postings=postings, doc_lengths=doc_lengths, doc_meta=doc_meta)

    phrase = Phrase(("event", "loop"))
    matches = phrase.evaluate(index)

    assert matches == {0}


def test_query_tree_evaluate_end_to_end() -> None:
    postings = {
        "python": [Posting(doc_id=0, tf=1), Posting(doc_id=1, tf=1)],
        "async": [Posting(doc_id=0, tf=1)],
        "java": [Posting(doc_id=1, tf=1)],
    }
    doc_lengths = {0: 10, 1: 10}
    doc_meta = {
        0: DocMeta(doc_id=0, path="d0.txt", title="Python Doc", length=10),
        1: DocMeta(doc_id=1, path="d1.txt", title="Java Doc", length=10),
    }
    index = Index(postings=postings, doc_lengths=doc_lengths, doc_meta=doc_meta)

    query = parse_query("python NOT java")
    matched = query.evaluate(index)
    assert matched == {0}
