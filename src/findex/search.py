import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from findex.index import InvertedIndex, Posting
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Search index using Boolean queries.")
    parser.add_argument("index_path", type=Path, help="Path to saved index file")
    parser.add_argument("query", type=str, help="Search query string")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge", help="Query engine")
    parser.add_argument("--limit", type=int, default=10, help="Max results to display")
    args = parser.parse_args()

    from findex.store import load_index

    start_load = time.perf_counter()
    index = load_index(args.index_path)
    load_time = time.perf_counter() - start_load

    start_query = time.perf_counter()
    matched_ids = execute_boolean_query(index, args.query, engine=args.engine)
    query_time = (time.perf_counter() - start_query) * 1000

    print(f"Index loaded in {load_time:.3f} s ({index.num_documents} docs, {index.num_terms} terms)")
    print(f"Query: '{args.query}' | Engine: {args.engine}")
    print(f"Found {len(matched_ids)} matching documents in {query_time:.2f} ms\n")

    for i, doc_id in enumerate(matched_ids[: args.limit]):
        meta = index.doc_meta.get(doc_id)
        if meta:
            print(f"[{i + 1}] (ID: {doc_id}) {meta.title} — {meta.path}")
        else:
            print(f"[{i + 1}] Document ID: {doc_id}")

    if len(matched_ids) > args.limit:
        print(f"... and {len(matched_ids) - args.limit} more.")


if __name__ == "__main__":
    main()
