import argparse
import functools
import time
from collections.abc import Sequence
from pathlib import Path

from findex.models import Index, InvertedIndex, Posting
from findex.query import parse_query
from findex.scoring import BM25, Scorer, SearchResult, TfIdf, ranked_search
from findex.snippet import snippet_for_doc
from findex.timing import timed
from findex.tokenize import tokenize


def merge_and(a: Sequence[int], b: Sequence[int]) -> list[int]:
    result: list[int] = []
    i = 0
    j = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        val_a = a[i]
        val_b = b[j]
        if val_a == val_b:
            result.append(val_a)
            i += 1
            j += 1
        elif val_a < val_b:
            i += 1
        else:
            j += 1
    return result


def merge_or(a: Sequence[int], b: Sequence[int]) -> list[int]:
    result: list[int] = []
    i = 0
    j = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        val_a = a[i]
        val_b = b[j]
        if val_a == val_b:
            result.append(val_a)
            i += 1
            j += 1
        elif val_a < val_b:
            result.append(val_a)
            i += 1
        else:
            result.append(val_b)
            j += 1
    while i < len_a:
        result.append(a[i])
        i += 1
    while j < len_b:
        result.append(b[j])
        j += 1
    return result


def merge_diff(a: Sequence[int], b: Sequence[int]) -> list[int]:
    result: list[int] = []
    i = 0
    j = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        val_a = a[i]
        val_b = b[j]
        if val_a == val_b:
            i += 1
            j += 1
        elif val_a < val_b:
            result.append(val_a)
            i += 1
        else:
            j += 1
    while i < len_a:
        result.append(a[i])
        i += 1
    return result


def set_and(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) & set(b))


def set_or(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) | set(b))


def set_diff(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) - set(b))


def get_postings_doc_ids(postings: list[Posting] | None) -> list[int]:
    if not postings:
        return []
    return [p.doc_id for p in postings]


def execute_boolean_query(
    index: InvertedIndex,
    query: str,
    engine: str = "merge",
) -> list[int]:
    raw_tokens = query.strip().split()
    if not raw_tokens:
        return []

    all_doc_ids = sorted(index.doc_meta.keys())
    current_docs: list[int] | None = None
    next_op = "AND"
    negate_next = False

    for raw in raw_tokens:
        token_upper = raw.upper()
        if token_upper == "AND":
            next_op = "AND"
            continue
        if token_upper == "OR":
            next_op = "OR"
            continue
        if token_upper == "NOT":
            negate_next = True
            continue

        normalized_terms = list(tokenize(raw))
        if not normalized_terms:
            continue

        for term in normalized_terms:
            term_docs = get_postings_doc_ids(index.postings.get(term))

            if negate_next:
                if engine == "merge":
                    term_docs = merge_diff(all_doc_ids, term_docs)
                else:
                    term_docs = set_diff(all_doc_ids, term_docs)
                negate_next = False

            if current_docs is None:
                current_docs = term_docs
            else:
                if next_op == "AND":
                    if engine == "merge":
                        current_docs = merge_and(current_docs, term_docs)
                    else:
                        current_docs = set_and(current_docs, term_docs)
                elif next_op == "OR":
                    if engine == "merge":
                        current_docs = merge_or(current_docs, term_docs)
                    else:
                        current_docs = set_or(current_docs, term_docs)

            next_op = "AND"

    return current_docs if current_docs is not None else []


@functools.lru_cache(maxsize=256)
def _cached_search_doc_ids(query: str, index: Index) -> tuple[int, ...]:
    query_tree = parse_query(query)
    matched_set = query_tree.evaluate(index)
    return tuple(sorted(matched_set))


@timed
def search(
    query: str,
    index: Index,
    scorer: Scorer | None = None,
    k: int = 10,
) -> list[SearchResult]:
    matched_tuple = _cached_search_doc_ids(query, index)
    if not matched_tuple:
        return []

    matched_set = set(matched_tuple)
    return ranked_search(index, query, scorer=scorer, k=k, matched_doc_ids=matched_set)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search index using Boolean or ranked queries.")
    parser.add_argument("index_path", type=Path, help="Path to saved index file")
    parser.add_argument("query", type=str, help="Search query string")
    parser.add_argument(
        "--scorer",
        choices=["bm25", "tfidf", "boolean"],
        default="bm25",
        help="Ranking scorer",
    )
    parser.add_argument("--engine", choices=["merge", "set"], default="merge", help="Legacy query engine")
    parser.add_argument("--limit", type=int, default=10, help="Max results to display")
    parser.add_argument("--snippets", action="store_true", help="Display snippet highlights")
    args = parser.parse_args()

    from findex.store import load_index

    start_load = time.perf_counter()
    index = load_index(args.index_path)
    load_time = time.perf_counter() - start_load

    print(f"Index loaded in {load_time:.3f} s ({index.num_documents} docs, {index.num_terms} terms)")

    if args.scorer == "boolean":
        start_query = time.perf_counter()
        matched_ids = execute_boolean_query(index, args.query, engine=args.engine)
        query_time = (time.perf_counter() - start_query) * 1000

        print(f"Query: '{args.query}' | Engine: {args.engine}")
        print(f"Found {len(matched_ids)} matching documents in {query_time:.2f} ms\n")

        for i, doc_id in enumerate(matched_ids[: args.limit]):
            meta = index.doc_meta.get(doc_id)
            if meta:
                print(f"[{i + 1}] (ID: {doc_id}) {meta.title} — {meta.path}")
            else:
                print(f"[{i + 1}] Document ID: {doc_id}")
        return

    active_scorer = BM25() if args.scorer == "bm25" else TfIdf()
    start_search = time.perf_counter()
    results = search(args.query, index, scorer=active_scorer, k=args.limit)
    elapsed_ms = (time.perf_counter() - start_search) * 1000.0

    print(f"Query: '{args.query}' | Scorer: {args.scorer.upper()} | Found {len(results)} results in {elapsed_ms:.2f} ms\n")

    query_terms = list(tokenize(args.query))
    for i, res in enumerate(results):
        print(f"[{i + 1}] (Score: {res.score:.4f}, ID: {res.doc_id}) {res.title} — {res.path}")
        if args.snippets:
            snip = snippet_for_doc(res.path, query_terms)
            if snip:
                print(f"    Snippet: {snip}")


if __name__ == "__main__":
    main()
