import json
from pathlib import Path

from typer.testing import CliRunner

from findex.cli import app
from findex.models import Index

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "full-text search engine" in result.stdout.lower()


def test_cli_index_not_found() -> None:
    result = runner.invoke(app, ["index", "non_existent_corpus_dir"])
    assert result.exit_code == 1
    assert "error" in result.stderr.lower()


def test_cli_index_and_stats(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.txt").write_text("Hello world in corpus.", encoding="utf-8")
    (corpus_dir / "doc2.txt").write_text("Another document with hello.", encoding="utf-8")

    out_file = tmp_path / "index.pkl"
    res_index = runner.invoke(app, ["index", str(corpus_dir), "--out", str(out_file)])
    assert res_index.exit_code == 0
    assert "successfully indexed" in res_index.stdout.lower()
    assert out_file.exists()

    res_stats = runner.invoke(app, ["stats", str(out_file)])
    assert res_stats.exit_code == 0
    assert "index statistics" in res_stats.stdout.lower()
    assert "total documents" in res_stats.stdout.lower()


def test_cli_search_json_and_table(tmp_path: Path, session_index: Index) -> None:
    from findex.store import save_index

    index_path = tmp_path / "session_index.pkl"
    save_index(session_index, index_path)

    res_search = runner.invoke(app, ["search-cmd", str(index_path), "alice", "--scorer", "bm25"])
    assert res_search.exit_code == 0
    assert "search results for" in res_search.stdout.lower()
    assert "doc2" in res_search.stdout.lower()

    res_json = runner.invoke(app, ["search-cmd", str(index_path), "alice", "--json"])
    assert res_json.exit_code == 0
    lines = [line.strip() for line in res_json.stdout.splitlines() if line.strip()]
    assert len(lines) > 0
    parsed = json.loads(lines[0])
    assert "doc_id" in parsed
    assert "score" in parsed
    assert "title" in parsed
