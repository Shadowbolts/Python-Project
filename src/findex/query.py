from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from findex.models import Index
from findex.tokenize import tokenize


class QueryNode(ABC):
    @abstractmethod
    def evaluate(self, index: Index) -> set[int]: ...

    def __and__(self, other: QueryNode) -> QueryNode:
        return And(self, other)

    def __or__(self, other: QueryNode) -> QueryNode:
        return Or(self, other)

    def __invert__(self) -> QueryNode:
        return Not(self)


@dataclass(frozen=True, slots=True)
class Term(QueryNode):
    term: str

    def evaluate(self, index: Index) -> set[int]:
        normalized = list(tokenize(self.term))
        if not normalized:
            return set()
        cleaned = normalized[0]
        if cleaned in index:
            return {p.doc_id for p in index[cleaned]}
        return set()


@dataclass(frozen=True, slots=True)
class Phrase(QueryNode):
    terms: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        if not self.terms:
            return set()

        cleaned_terms: list[str] = []
        for t in self.terms:
            toks = list(tokenize(t))
            if toks:
                cleaned_terms.append(toks[0])

        if not cleaned_terms:
            return set()

        if len(cleaned_terms) == 1:
            return Term(cleaned_terms[0]).evaluate(index)

        first_term = cleaned_terms[0]
        if first_term not in index:
            return set()

        candidate_docs: dict[int, list[int]] = {
            p.doc_id: list(p.positions) for p in index[first_term] if p.positions
        }

        for offset, next_term in enumerate(cleaned_terms[1:], start=1):
            if next_term not in index:
                return set()
            next_postings = {
                p.doc_id: set(p.positions) for p in index[next_term] if p.positions
            }

            matched_docs: dict[int, list[int]] = {}
            for doc_id, base_positions in candidate_docs.items():
                if doc_id in next_postings:
                    valid_positions = [
                        pos for pos in base_positions if (pos + offset) in next_postings[doc_id]
                    ]
                    if valid_positions:
                        matched_docs[doc_id] = valid_positions

            candidate_docs = matched_docs
            if not candidate_docs:
                break

        return set(candidate_docs.keys())


@dataclass(frozen=True, slots=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)


@dataclass(frozen=True, slots=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)


@dataclass(frozen=True, slots=True)
class Not(QueryNode):
    operand: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        all_doc_ids = set(index.doc_meta.keys())
        return all_doc_ids - self.operand.evaluate(index)


TOKEN_SPEC = [
    ("PHRASE", r'"[^"]*"'),
    ("AND", r"\bAND\b"),
    ("OR", r"\bOR\b"),
    ("NOT", r"\bNOT\b"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("TERM", r'[^\s()"]+'),
]
TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPEC))


@dataclass(frozen=True, slots=True)
class Token:
    type: str
    value: str


def tokenize_query(query: str) -> list[Token]:
    tokens: list[Token] = []
    for match in TOKEN_RE.finditer(query):
        kind = match.lastgroup
        if kind is None:
            continue
        val = match.group(0)
        tokens.append(Token(type=kind, value=val))
    return tokens


class QueryParser:
    def __init__(self, tokens: Sequence[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def match(self, token_type: str) -> Token | None:
        token = self.peek()
        if token and token.type == token_type:
            self.pos += 1
            return token
        return None

    def parse(self) -> QueryNode:
        if not self.tokens:
            return Term("")
        return self.parse_or()

    def parse_or(self) -> QueryNode:
        node = self.parse_and()
        while self.match("OR"):
            right = self.parse_and()
            node = Or(node, right)
        return node

    def parse_and(self) -> QueryNode:
        node = self.parse_not()
        while True:
            peek_tok = self.peek()
            if self.match("AND") or (peek_tok is not None and peek_tok.type in ("TERM", "PHRASE", "LPAREN", "NOT")):
                right = self.parse_not()
                node = And(node, right)
            else:
                break
        return node

    def parse_not(self) -> QueryNode:
        if self.match("NOT"):
            operand = self.parse_not()
            return Not(operand)
        return self.parse_atom()

    def parse_atom(self) -> QueryNode:
        if self.match("LPAREN"):
            node = self.parse_or()
            self.match("RPAREN")
            return node

        phrase_tok = self.match("PHRASE")
        if phrase_tok:
            content = phrase_tok.value.strip('"')
            terms = tuple(tokenize(content))
            return Phrase(terms)

        term_tok = self.match("TERM")
        if term_tok:
            norm_terms = list(tokenize(term_tok.value))
            if len(norm_terms) > 1:
                return Phrase(tuple(norm_terms))
            term_str = norm_terms[0] if norm_terms else term_tok.value
            return Term(term_str)

        return Term("")


def parse_query(query: str) -> QueryNode:
    tokens = tokenize_query(query)
    parser = QueryParser(tokens)
    return parser.parse()
