"""Splits generated (or reference) text into atomic assertions.

Section 5.9 of the paper evaluates this step ("Assertion Extraction and
Verification Evaluation") independently of end-to-end generation,
against an "Automatic assertions" vs. "Gold assertions" comparison
(Table 4), precisely to separate assertion-extraction error from graph
-matching/verification error (RQ6).

The default splitter is clause-based (sentence + coordinating
conjunction splitting), which is a reasonable, dependency-free
approximation of "one clinical claim per assertion". For the numbers
reported in the paper, consider replacing this with an
OpenIE-style or LLM-based atomic-fact decomposer and documenting the
choice in `configs/default.yaml`.
"""
from __future__ import annotations

import re

from nsrag.schema import Assertion

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT = re.compile(r"\s*,?\s+\b(and|but|while)\b\s+", re.IGNORECASE)


def extract_assertions(text: str, id_prefix: str = "a") -> list[Assertion]:
    assertions: list[Assertion] = []
    idx = 0
    for sent in _SENT_SPLIT.split(text.strip()):
        sent = sent.strip()
        if not sent:
            continue
        for clause in _CLAUSE_SPLIT.split(sent):
            clause = clause.strip(" .")
            if clause and clause.lower() not in {"and", "but", "while"} and len(clause.split()) > 1:
                assertions.append(Assertion(id=f"{id_prefix}_{idx}", text=clause))
                idx += 1
    return assertions
