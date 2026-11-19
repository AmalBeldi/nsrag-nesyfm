"""Three-state symbolic verification (paper Section 3.4 / 5.8).

For each extracted assertion, checks it against the patient's
graph-grounded evidence (the retrieved triples C_{p,T}, or the full
graph if you want an "oracle" upper bound) and assigns one of:

  SUPPORTED    -- matching evidence found, no contradiction
  CONTRADICTED -- evidence found for the same subject/predicate with an
                  incompatible object, or an explicit negation mismatch
  UNKNOWN      -- neither sufficient supporting nor contradicting
                  evidence is available

This lexical-overlap + negation-cue implementation is a transparent,
fully inspectable baseline verifier. It is intentionally conservative
(a real deployment would likely add an NLI model as a second-stage
verifier for paraphrase-heavy assertions) -- swap in a stronger matcher
behind the same `Verifier` interface without touching the rest of the
pipeline, and report which one you used in the paper.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from nsrag.schema import Assertion, Triple, VerificationState

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_NEGATION_CUES = {
    "no", "not", "denies", "denied", "negative", "without", "absence", "ruled out",
}
# Stripped only for verification overlap scoring: generic determiners
# and the (near-universal) "patient" subject would otherwise dilute
# the Jaccard signal on the clinically meaningful terms.
_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "be", "been", "of", "on", "in", "to",
    "and", "with", "for", "patient", "this",
}


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)} - _STOPWORDS


def _numbers(text: str) -> set[str]:
    """Numeric tokens (digits only). A support match is rejected when both
    sides carry numbers and the numbers differ (e.g. HbA1c 8.2 vs 6.0):
    shared unit/analyte tokens must not mask a value mismatch."""
    return {w for w in _WORD_RE.findall(text) if w.isdigit()}


def _is_negated(text: str) -> bool:
    tokens = _tokenize(text)
    return bool(tokens & _NEGATION_CUES)


@dataclass
class VerifierConfig:
    support_threshold: float = 0.5    # min Jaccard overlap to count as a match
    contradiction_margin: float = 0.3  # min overlap on subject/predicate to flag a conflict
    # Open-world assumption: absence of evidence => UNKNOWN. The
    # same-predicate/different-object contradiction rule is only valid for
    # *functional* predicates (one value at a time, e.g. a lab result).
    # For multi-valued predicates (has_condition, takes_medication) a
    # different object is NOT a conflict. Predicates are matched by prefix.
    functional_predicate_prefixes: tuple = ("lab_",)


class GraphVerifier:
    def __init__(self, config: VerifierConfig | None = None):
        self.config = config or VerifierConfig()

    def _overlap(self, a: str, b: str) -> float:
        ta, tb = _tokenize(a), _tokenize(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def verify(self, assertion: Assertion, evidence: list[Triple]) -> Assertion:
        assertion_negated = _is_negated(assertion.text)

        scored = [(t, self._overlap(assertion.text, t.as_text())) for t in evidence]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Negation cues eat into the assertion's own token budget, which
        # lowers raw Jaccard overlap even for a clean match; apply a
        # relaxed threshold when checking for a negation mismatch so a
        # denied finding can still be flagged as CONTRADICTED against
        # matching positive evidence.
        negation_threshold = self.config.support_threshold * 0.75
        if scored and scored[0][1] >= negation_threshold:
            best_triple, best_score = scored[0]
            triple_negated = _is_negated(best_triple.as_text())
            if assertion_negated != triple_negated:
                assertion.state = VerificationState.CONTRADICTED
                assertion.confidence = round(best_score, 3)
                assertion.witness_subgraph = [best_triple]
                return assertion

        a_nums = _numbers(assertion.text)
        if scored and scored[0][1] >= self.config.support_threshold and not (
                a_nums and _numbers(scored[0][0].as_text()) and a_nums != _numbers(scored[0][0].as_text())):
            best_score = scored[0][1]
            assertion.state = VerificationState.SUPPORTED
            assertion.confidence = round(best_score, 3)
            assertion.witness_subgraph = [t for t, s in scored if s >= self.config.support_threshold]
            return assertion

        # No strong match: look for a same-predicate conflict (weaker
        # overlap, but on the semantic "core" of the claim -- the
        # predicate, e.g. "has_condition" -- rather than the often
        # generic subject "Patient") as a secondary contradiction
        # signal.
        assertion_terms = _tokenize(assertion.text)
        for triple, score in scored:
            if not triple.pred.lower().startswith(self.config.functional_predicate_prefixes):
                continue  # multi-valued predicate: different object != contradiction
            pred_terms = _tokenize(triple.pred.replace("_", " "))
            if not pred_terms:
                continue
            pred_overlap = len(pred_terms & assertion_terms) / len(pred_terms)
            if pred_overlap >= 1.0:  # functional: the analyte itself must be named
                obj_terms = _tokenize(triple.obj)
                if not (obj_terms <= assertion_terms):  # functional: whole value must match
                    assertion.state = VerificationState.CONTRADICTED
                    assertion.confidence = round(score, 3)
                    assertion.witness_subgraph = [triple]
                    return assertion

        assertion.state = VerificationState.UNKNOWN
        assertion.confidence = round(scored[0][1], 3) if scored else 0.0
        assertion.witness_subgraph = []
        return assertion

    def verify_all(self, assertions: list[Assertion], evidence: list[Triple]) -> list[Assertion]:
        return [self.verify(a, evidence) for a in assertions]
