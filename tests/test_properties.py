import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.models import DocMeta, Index, Posting
from findex.search import merge_and, set_and
from findex.store import load_index, save_index
from findex.tokenize import tokenize


@given(st.text())
def test_property_tokenize_idempotence_and_non_empty(text: str) -> None:
    tokens = list(tokenize(text))
    assert all(len(t) > 0 for t in tokens)
    retokenized = list(tokenize(" ".join(tokens)))
    assert retokenized == tokens


@given(
    st.lists(st.integers(min_value=0, max_value=1000), unique=True),
    st.lists(st.integers(min_value=0, max_value=1000), unique=True),
)
def test_property_merge_and_matches_set_oracle(a: list[int], b: list[int]) -> None:
    sorted_a = sorted(a)
    sorted_b = sorted(b)
    merge_res = merge_and(sorted_a, sorted_b)
    oracle_res = set_and(sorted_a, sorted_b)
    assert merge_res == oracle_res


@given(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
        values=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=50),
                st.integers(min_value=1, max_value=20),
            ).map(lambda t: Posting(doc_id=t[0], tf=t[1])),
            max_size=5,
        ),
        max_size=10,
    )
)
def test_property_save_load_roundtrip_json(tmp_path_factory: pytest.TempPathFactory, postings_data: dict[str, list[Posting]]) -> None:
    tmp_dir = tmp_path_factory.mktemp("rt")
    file_path = tmp_dir / "test.json"
    
    doc_meta = {0: DocMeta(doc_id=0, path="d.txt", title="D", length=10)}
    doc_lengths = {0: 10}
    index = Index(postings=postings_data, doc_lengths=doc_lengths, doc_meta=doc_meta)

    save_index(index, file_path, format="json")
    loaded = load_index(file_path, format="json")

    assert len(loaded) == len(index)
    assert set(loaded.keys()) == set(index.keys())
