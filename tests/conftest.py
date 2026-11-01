from pathlib import Path

import pytest

from findex.index import build_index
from findex.models import Index


@pytest.fixture(scope="session")
def fixture_corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    corpus_dir = tmp_path_factory.mktemp("corpus")
    doc1 = corpus_dir / "doc1.txt"
    doc2 = corpus_dir / "doc2.txt"
    doc3 = corpus_dir / "doc3.txt"

    doc1.write_text("The quick brown fox jumps over the lazy dog.", encoding="utf-8")
    doc2.write_text("Alice adventures in wonderland with rabbit and queen.", encoding="utf-8")
    doc3.write_text("Sherlock holmes and doctor watson baker street.", encoding="utf-8")
    return corpus_dir

@pytest.fixture(scope="session")
def session_index(fixture_corpus_dir: Path) -> Index:
    return build_index(fixture_corpus_dir, record_positions=True)
