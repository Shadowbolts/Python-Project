import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    if root.is_file():
        file_paths = iter([root])
    else:
        file_paths = root.rglob("*.txt")

    for path in file_paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning("Error reading %s: %s", path, e)
            continue

        doc_id = str(path.relative_to(root)) if not root.is_file() else path.name
        yield Document(doc_id=doc_id, path=path, text=text)
