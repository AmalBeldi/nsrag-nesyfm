"""Factuality and three-state verification metrics (paper Section 5.8).

All formulas here are transcribed directly from the paper so that the
code and the LaTeX are guaranteed to stay in sync:

    P_fact = TP / (TP + FP)              R_fact = TP / (TP + FN)
    F1_fact = 2 P_fact R_fact / (P_fact + R_fact)

    EC = |E_sup| / |E(A)|                 (Explanation Coverage)
    CR = |E_con| / |E(A)|                 (Contradiction Rate = Hallucination Rate)
    UR = |E_unk| / |E(A)|                 (Unknown Rate)
    EC + CR + UR = 1

    MR = |G_min| / |G_exp|                (Minimality Ratio, averaged over
                                            SUPPORTED assertions with non-empty G_exp)

    F1_macro (verification) = mean over {SUPPORTED, CONTRADICTED, UNKNOWN}
                               of per-class F1 against reference labels
"""
from __future__ import annotations

from dataclasses import dataclass

from nsrag.schema import Assertion, VerificationState


def _match(a_text: str, reference_texts: set[str], threshold: float = 0.5) -> bool:
    """Lexical-overlap matching between a generated assertion and the
    independently constructed reference assertions (E_ref, Section
    5.7). Swap for embedding-similarity matching if you want a softer
    match criterion -- keep the threshold documented in
    `configs/default.yaml` either way, since Fact P/R/F1 are sensitive
    to it.
    """
    a_tokens = set(a_text.lower().split())
    for ref in reference_texts:
        ref_tokens = set(ref.lower().split())
        if not a_tokens or not ref_tokens:
            continue
        jaccard = len(a_tokens & ref_tokens) / len(a_tokens | ref_tokens)
        if jaccard >= threshold:
            return True
    return False


@dataclass
class FactualityScore:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def factuality_score(
    generated_assertions: list[Assertion],
    reference_assertions: list[str],
    match_threshold: float = 0.5,
) -> FactualityScore:
    ref_set = set(reference_assertions)
    matched_refs: set[str] = set()

    tp = 0
    for a in generated_assertions:
        found = False
        for ref in ref_set - matched_refs:
            if _match(a.text, {ref}, match_threshold):
                matched_refs.add(ref)
                found = True
                break
        if found:
            tp += 1

    fp = len(generated_assertions) - tp
    fn = len(ref_set) - len(matched_refs)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return FactualityScore(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


@dataclass
class VerificationPartition:
    ec: float   # Explanation Coverage
    cr: float   # Contradiction Rate == Hallucination Rate
    ur: float   # Unknown Rate
    n_total: int
    n_supported: int
    n_contradicted: int
    n_unknown: int


def verification_partition(assertions: list[Assertion]) -> VerificationPartition:
    n = len(assertions)
    if n == 0:
        return VerificationPartition(0.0, 0.0, 0.0, 0, 0, 0, 0)

    n_sup = sum(1 for a in assertions if a.state == VerificationState.SUPPORTED)
    n_con = sum(1 for a in assertions if a.state == VerificationState.CONTRADICTED)
    n_unk = sum(1 for a in assertions if a.state == VerificationState.UNKNOWN)

    return VerificationPartition(
        ec=round(n_sup / n, 4),
        cr=round(n_con / n, 4),
        ur=round(n_unk / n, 4),
        n_total=n, n_supported=n_sup, n_contradicted=n_con, n_unknown=n_unk,
    )


def hallucination_rate(assertions: list[Assertion]) -> float:
    """HR is defined identically to CR (see main.tex, Section 5.8)."""
    return verification_partition(assertions).cr


def verification_macro_f1(
    predicted: list[VerificationState],
    reference: list[VerificationState],
) -> dict:
    """Class-wise Precision/Recall/F1 plus macro-F1 over the three
    verification states, computed against independently assigned
    reference verification labels (e.g. from human annotation on a
    validation subset)."""
    assert len(predicted) == len(reference), "predicted/reference length mismatch"
    classes = [VerificationState.SUPPORTED, VerificationState.CONTRADICTED, VerificationState.UNKNOWN]
    per_class = {}
    f1s = []
    for c in classes:
        tp = sum(1 for p, r in zip(predicted, reference) if p == c and r == c)
        fp = sum(1 for p, r in zip(predicted, reference) if p == c and r != c)
        fn = sum(1 for p, r in zip(predicted, reference) if p != c and r == c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[c.value] = {"precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
    return {"per_class": per_class, "macro_f1": round(sum(f1s) / len(f1s), 4)}


def minimality_ratio_corpus(mr_values: list[float]) -> float:
    """Corpus-level MR: mean of per-assertion MR over SUPPORTED
    assertions with a non-empty explanation graph (see
    `nsrag.verification.explain`)."""
    if not mr_values:
        return 0.0
    return round(sum(mr_values) / len(mr_values), 4)
