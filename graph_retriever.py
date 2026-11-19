"""Graph-grounded retrieval (paper Section 3.3).

Selects the top-k triples/paths from the (already task-summarized)
patient graph that are most relevant to the question, subject to a
maximum context-token budget B (Section 5.5, "Compared Methods":
`|C_m| <= B`). This is the retrieval mechanism specific to NS-RAG and
its "Graph Linearization" baseline (B3) reuses the same selection with
a flat textual serialization instead of a structured one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from nsrag.schema import Triple

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def _estimate_tokens(text: str) -> int:
    """Cheap whitespace-based token estimate. Swap for the actual
    tokenizer of your generation backbone (Section 5.6: Flan-T5-Large)
    if you need exact context-budget accounting."""
    return max(1, len(text.split()))


@dataclass
class RetrievalConfig:
    k: int = 25                 # paper default (Section 5.6)
    max_context_tokens: int = 2048  # paper default (Section 5.6)


class GraphRetriever:
    def __init__(self, config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()

    def score_triple(self, triple: Triple, query_terms: set[str]) -> float:
        triple_terms = _tokenize(triple.as_text())
        if not query_terms:
            return 0.0
        return len(triple_terms & query_terms) / len(query_terms)

    def retrieve(self, triples: list[Triple], question: str) -> list[Triple]:
        query_terms = _tokenize(question)
        scored = sorted(
            triples, key=lambda t: self.score_triple(t, query_terms), reverse=True
        )
        selected: list[Triple] = []
        budget = self.config.max_context_tokens
        for triple in scored[: self.config.k]:
            cost = _estimate_tokens(triple.as_text())
            if cost > budget:
                break
            selected.append(triple)
            budget -= cost
        return selected

    @staticmethod
    def to_structured_context(triples: list[Triple]) -> list[str]:
        """NS-RAG's structured evidence units: one triple per unit,
        kept separate (not flattened into a single string) so the
        generator and, later, the verifier can trace each generated
        assertion back to a specific evidence unit."""
        return [t.as_text() for t in triples]

    @staticmethod
    def to_linearized_context(triples: list[Triple]) -> str:
        """B3 (Graph Linearization) baseline: same retrieved evidence,
        flattened into a single text block with no structure preserved
        beyond triple boundaries."""
        return ". ".join(t.as_text() for t in triples) + "."
