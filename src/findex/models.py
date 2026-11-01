from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str
    length: int


@dataclass
class Index(Mapping[str, list[Posting]]):
    postings: dict[str, list[Posting]] = field(default_factory=dict)
    doc_lengths: dict[int, int] = field(default_factory=dict)
    doc_meta: dict[int, DocMeta] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.postings)

    def __contains__(self, term: object) -> bool:
        return term in self.postings

    def __getitem__(self, term: str) -> list[Posting]:
        return self.postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __repr__(self) -> str:
        avg_len = f"{self.avg_doc_length:.1f}" if self.num_docs > 0 else "0.0"
        return f"Index(terms={len(self)}, docs={self.num_docs}, avg_doc_len={avg_len})"

    @property
    def num_docs(self) -> int:
        return len(self.doc_meta)

    @property
    def num_documents(self) -> int:
        return self.num_docs

    @property
    def num_terms(self) -> int:
        return len(self)

    @cached_property
    def avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        postings = self.postings.get(term)
        return len(postings) if postings else 0


InvertedIndex = Index
