"""Minimal explanation subgraph extraction (paper Section 3.4, MR metric).

Given a set of witness triples for a SUPPORTED assertion (`G_exp`, the
union of everything the verifier matched against), reduces it to a
minimal sufficient subset `G_min`: the smallest subset of triples that
still passes the verifier for the same assertion. This directly
operationalizes the Minimality Ratio defined in Section 5.8:

    MR = |G_min| / |G_exp|

A witness triple is kept only if removing it would cause the assertion
to fail verification -- i.e. we greedily drop redundant triples and
check the invariant still holds. This is exact for the overlap-based
verifier in `verifier.py` and gives a reproducible upper bound on
explanation compactness for any (possibly stronger) verifier that is
monotonic in evidence (more evidence never hurts support).
"""
from __future__ import annotations

from nsrag.schema import Assertion, Triple
from nsrag.verification.verifier import GraphVerifier, VerificationState


def extract_minimal_explanation(
    assertion: Assertion,
    exp_graph: list[Triple],
    verifier: GraphVerifier,
) -> list[Triple]:
    if not exp_graph:
        return []

    kept = list(exp_graph)
    changed = True
    while changed and len(kept) > 1:
        changed = False
        for triple in list(kept):
            candidate = [t for t in kept if t is not triple]
            probe = Assertion(id=assertion.id + "_probe", text=assertion.text)
            verified = verifier.verify(probe, candidate)
            if verified.state == VerificationState.SUPPORTED:
                kept = candidate
                changed = True
                break
    return kept


def minimality_ratio(exp_graph: list[Triple], min_graph: list[Triple]) -> float:
    if not exp_graph:
        return 0.0
    return round(len(min_graph) / len(exp_graph), 3)
