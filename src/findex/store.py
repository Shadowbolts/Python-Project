import json
import pickle
from pathlib import Path

from findex.models import DocMeta, InvertedIndex, Posting


def save_index(index: InvertedIndex, path: Path, format: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = format or ("json" if path.suffix == ".json" else "pickle")

    if fmt == "pickle":
        with path.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif fmt == "json":
        data = {
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


def load_index(path: Path, format: str | None = None) -> InvertedIndex:
    fmt = format or ("json" if path.suffix == ".json" else "pickle")

    if fmt == "pickle":
        with path.open("rb") as f:
            obj = pickle.load(f)
            if not isinstance(obj, InvertedIndex):
                msg = "Loaded object is not an InvertedIndex"
                raise TypeError(msg)
            return obj
    elif fmt == "json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        postings = {
            term: [
                Posting(
                    doc_id=raw[0],
                    tf=raw[1],
                    positions=tuple(raw[2]) if len(raw) > 2 else (),
                )
                for raw in plist
            ]
            for term, plist in data["postings"].items()
        }
        doc_lengths = {int(k): v for k, v in data["doc_lengths"].items()}
        doc_meta = {
            int(k): DocMeta(
                doc_id=int(k),
                path=v["path"],
                title=v["title"],
                length=v["length"],
            )
            for k, v in data["doc_meta"].items()
        }
        return InvertedIndex(postings=postings, doc_lengths=doc_lengths, doc_meta=doc_meta)
    else:
        msg = f"Unsupported format: {fmt}"
        raise ValueError(msg)


save = save_index
load = load_index
