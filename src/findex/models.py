from dataclasses import dataclass, field


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
class InvertedIndex:
    postings: dict[str, list[Posting]] = field(default_factory=dict)
    doc_lengths: dict[int, int] = field(default_factory=dict)
    doc_meta: dict[int, DocMeta] = field(default_factory=dict)

    @property
    def num_documents(self) -> int:
        return len(self.doc_meta)

    @property
    def num_terms(self) -> int:
        return len(self.postings)
