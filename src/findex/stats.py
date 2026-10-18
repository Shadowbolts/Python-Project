import argparse
import itertools
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize


def compute_stats_lazy(
    docs: Iterator[Document],
) -> tuple[int, int, int, list[tuple[str, int]]]:
    doc_count = 0
    total_tokens = 0
    counter: Counter[str] = Counter()

    for doc in docs:
        doc_count += 1
        for token in tokenize(doc.text):
            total_tokens += 1
            counter[token] += 1

    vocab_size = len(counter)
    top_50 = counter.most_common(50)
    return doc_count, total_tokens, vocab_size, top_50


def compute_stats_eager(
    docs: Iterator[Document],
) -> tuple[int, int, int, list[tuple[str, int]]]:
    all_docs = list(docs)
    doc_count = len(all_docs)
    all_tokens = [list(tokenize(doc.text)) for doc in all_docs]
    flat_tokens = [tok for doc_toks in all_tokens for tok in doc_toks]
    total_tokens = len(flat_tokens)
    counter: Counter[str] = Counter(flat_tokens)
    vocab_size = len(counter)
    top_50 = counter.most_common(50)
    return doc_count, total_tokens, vocab_size, top_50


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus statistics pipeline")
    parser.add_argument("path", type=Path, help="Path to corpus directory or file")
    parser.add_argument(
        "--limit", "-n", type=int, default=None, help="Limit number of documents"
    )
    parser.add_argument(
        "--eager", action="store_true", help="Run eager version instead of lazy"
    )
    args = parser.parse_args()

    docs = iter_documents(args.path)
    if args.limit is not None:
        docs = itertools.islice(docs, args.limit)

    tracemalloc.start()
    start_time = time.perf_counter()

    if args.eager:
        doc_count, total_tokens, vocab_size, top_terms = compute_stats_eager(docs)
    else:
        doc_count, total_tokens, vocab_size, top_terms = compute_stats_lazy(docs)

    elapsed = time.perf_counter() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Mode:            {'eager' if args.eager else 'lazy'}")
    print(f"Documents:       {doc_count}")
    print(f"Total tokens:    {total_tokens}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Elapsed time:    {elapsed:.3f} s")
    print(f"Peak memory:     {peak_mem / (1024 * 1024):.2f} MB")
    print("\nTop 50 terms:")
    for rank, (term, count) in enumerate(top_terms, 1):
        print(f"{rank:2d}. {term:<20} {count}")


if __name__ == "__main__":
    main()
