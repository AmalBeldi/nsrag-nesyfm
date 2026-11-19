"""Textual retrieval baseline (B2 -- Textual RAG, paper Section 5.4).

Retrieves raw text spans (sentences/chunks) from the patient's clinical
notes using TF-IDF cosine similarity, discarding the relational and
temporal structure that NS-RAG preserves. This is the paper's key
contrastive baseline for RQ1.

Uses scikit-learn's TfidfVectorizer (already a project dependency) so
this baseline runs anywhere with no additional models or GPU, exactly
as described for B2 in the paper.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_into_chunks(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


@dataclass
class TextRetrievalConfig:
    top_k: int = 25
    max_context_tokens: int = 2048


class TextRetriever:
    """Fits a fresh TF-IDF index per patient (patient notes are the only
    valid retrieval scope -- see Section 5.6, no cross-patient leakage)."""

    def __init__(self, config: TextRetrievalConfig | None = None):
        self.config = config or TextRetrievalConfig()

    def retrieve(self, note_texts: list[str], question: str) -> list[str]:
        chunks: list[str] = []
        for text in note_texts:
            chunks.extend(split_into_chunks(text))
        if not chunks:
            return []

        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            matrix = vectorizer.fit_transform(chunks + [question])
        except ValueError:
            # Degenerate case: empty vocabulary (e.g. all-stopword chunks).
            return chunks[: self.config.top_k]

        chunk_vecs, query_vec = matrix[:-1], matrix[-1]
        sims = cosine_similarity(chunk_vecs, query_vec).ravel()
        order = sims.argsort()[::-1]

        selected: list[str] = []
        budget = self.config.max_context_tokens
        for idx in order[: self.config.top_k]:
            chunk = chunks[idx]
            cost = max(1, len(chunk.split()))
            if cost > budget:
                break
            selected.append(chunk)
            budget -= cost
        return selected
