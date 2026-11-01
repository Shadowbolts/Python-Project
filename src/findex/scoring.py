import heapq
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from findex.models import Index, Posting
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True, order=True)
class SearchResult:
    score: float
    doc_id: int
    title: str
    path: str


@runtime_checkable
class Scorer(Protocol):
    def score(self, term: str, posting: Posting, index: Index) -> float: ...

    def __call__(self, term: str, posting: Posting, index: Index) -> float: ...


class TfIdf:
    def score(self, term: str, posting: Posting, index: Index) -> float:
        tf_weight = 1.0 + math.log(posting.tf) if posting.tf > 0 else 0.0
        doc_count = index.num_docs
        df = index.df(term)
        idf = math.log((doc_count + 1.0) / (df + 1.0)) + 1.0
        return tf_weight * idf

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(self, term: str, posting: Posting, index: Index) -> float:
        doc_count = index.num_docs
        df = index.df(term)
        idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1.0)

        doc_len = index.doc_length(posting.doc_id)
        avg_len = index.avg_doc_length or 1.0

        len_norm = 1.0 - self.b + self.b * (doc_len / avg_len)
        tf_component = (posting.tf * (self.k1 + 1.0)) / (posting.tf + self.k1 * len_norm)

        return idf * tf_component

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)


def ranked_search(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    k: int = 10,
    matched_doc_ids: set[int] | None = None,
) -> list[SearchResult]:
    active_scorer = scorer or BM25()
    tokens = list(tokenize(query))
    if not tokens:
        return []

    accumulated_scores: defaultdict[int, float] = defaultdict(float)

    for token in tokens:
        if token not in index:
            continue
        for posting in index[token]:
            if matched_doc_ids is not None and posting.doc_id not in matched_doc_ids:
                continue
            accumulated_scores[posting.doc_id] += active_scorer.score(token, posting, index)

    if not accumulated_scores:
        return []

    top_doc_scores = heapq.nlargest(k, accumulated_scores.items(), key=lambda item: item[1])

    results: list[SearchResult] = []
    for doc_id, score in top_doc_scores:
        meta = index.doc_meta[doc_id]
        results.append(
            SearchResult(
                score=score,
                doc_id=doc_id,
                title=meta.title,
                path=meta.path,
            )
        )

    return results
