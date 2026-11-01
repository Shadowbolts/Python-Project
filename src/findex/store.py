import json
import pickle
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast

from findex.models import DocMeta, Index, InvertedIndex, Posting
from findex.timing import timed


def save_index(
    index: Index,
    path: Path,
    format: Literal["pickle", "json"] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = format or ("json" if path.suffix == ".json" else "pickle")

    if fmt == "pickle":
        with path.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif fmt == "json":
        data: dict[str, Any] = {
            "postings": {
                term: [
                    [p.doc_id, p.tf, list(p.positions)]
                    for p in plist
                ]
                for term, plist in index.postings.items()
            },
            "doc_lengths": {str(k): v for k, v in index.doc_lengths.items()},
            "doc_meta": {
                str(k): {
                    "path": v.path,
                    "title": v.title,
                    "length": v.length,
                }
                for k, v in index.doc_meta.items()
            },
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
    else:
        msg = f"Unsupported format: {fmt}"
        raise ValueError(msg)


@timed
def load_index(
    path: Path,
    format: Literal["pickle", "json"] | None = None,
) -> Index:
    fmt = format or ("json" if path.suffix == ".json" else "pickle")

    if fmt == "pickle":
        with path.open("rb") as f:
            obj: object = pickle.load(f)
            if not isinstance(obj, (Index, InvertedIndex)):
                msg = "Loaded object is not an Index"
                raise TypeError(msg)
            return obj
    elif fmt == "json":
        with path.open("r", encoding="utf-8") as f:
            raw_data = cast(dict[str, Any], json.load(f))

        raw_postings = cast(dict[str, list[list[Any]]], raw_data["postings"])
        postings: dict[str, list[Posting]] = {}
        for term, plist in raw_postings.items():
            item_list: list[Posting] = []
            for raw in plist:
                positions: tuple[int, ...] = (
                    tuple(int(x) for x in raw[2]) if len(raw) > 2 else ()
                )
                item_list.append(
                    Posting(
                        doc_id=int(raw[0]),
                        tf=int(raw[1]),
                        positions=positions,
                    )
                )
            postings[term] = item_list

        raw_lengths = cast(dict[str, int], raw_data["doc_lengths"])
        doc_lengths: dict[int, int] = {int(k): int(v) for k, v in raw_lengths.items()}

        raw_meta = cast(dict[str, dict[str, Any]], raw_data["doc_meta"])
        doc_meta: dict[int, DocMeta] = {}
        for k, v in raw_meta.items():
            doc_meta[int(k)] = DocMeta(
                doc_id=int(k),
                path=str(v["path"]),
                title=str(v["title"]),
                length=int(v["length"]),
            )

        return Index(postings=postings, doc_lengths=doc_lengths, doc_meta=doc_meta)
    else:
        msg = f"Unsupported format: {fmt}"
        raise ValueError(msg)


@contextmanager
def open_index(
    path: Path,
    format: Literal["pickle", "json"] | None = None,
) -> Generator[Index, None, None]:
    index = load_index(path, format=format)
    try:
        yield index
    finally:
        pass


save = save_index
load = load_index
