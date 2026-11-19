"""22-probe suite for the semantics of the three-state verifier.

A constructed record (no patient data) with typed predicates; expected
states were fixed before running the verifier but AFTER the closed-world
behaviour of an earlier version had been observed. The verifier was later
tuned on these same probes, so the score is a regression suite, not a
held-out estimate. Usage: PYTHONPATH=src python scripts/verification_probes.py
"""
from collections import Counter
from nsrag.schema import Assertion, Triple
from nsrag.verification.verifier import GraphVerifier

EVIDENCE = [
    Triple("Patient", "has_condition", "type 2 diabetes mellitus"),
    Triple("Patient", "has_condition", "hypertension"),
    Triple("Patient", "takes_medication", "metformin 500 mg"),
    Triple("Patient", "takes_medication", "lisinopril 10 mg"),
    Triple("Patient", "lab_HbA1c", "8.2 percent"),
    Triple("Patient", "lab_creatinine", "1.1 mg/dl"),
]
PROBES = [
    ("supported-exact", "patient has type 2 diabetes mellitus", "SUPPORTED"),
    ("supported-exact", "patient has hypertension", "SUPPORTED"),
    ("supported-exact", "patient takes metformin 500 mg", "SUPPORTED"),
    ("supported-exact", "patient takes lisinopril 10 mg", "SUPPORTED"),
    ("supported-exact", "patient lab HbA1c 8.2 percent", "SUPPORTED"),
    ("absent-condition", "patient has asthma", "UNKNOWN"),
    ("absent-condition", "patient has chronic kidney disease", "UNKNOWN"),
    ("absent-condition", "patient has heart failure", "UNKNOWN"),
    ("absent-medication", "patient takes insulin glargine", "UNKNOWN"),
    ("absent-medication", "patient takes warfarin 5 mg", "UNKNOWN"),
    ("absent-medication", "patient takes atorvastatin 20 mg", "UNKNOWN"),
    ("negated-present", "patient has no hypertension", "CONTRADICTED"),
    ("negated-present", "patient denies type 2 diabetes mellitus", "CONTRADICTED"),
    ("negated-present", "patient does not take metformin 500 mg", "CONTRADICTED"),
    ("negated-absent", "patient has no asthma", "UNKNOWN"),
    ("negated-absent", "patient does not take warfarin 5 mg", "UNKNOWN"),
    ("lab-conflict", "patient lab HbA1c 6.0 percent", "CONTRADICTED"),
    ("lab-conflict", "patient lab creatinine 2.4 mg/dl", "CONTRADICTED"),
    ("lab-absent", "patient lab sodium 140 mmol/l", "UNKNOWN"),
    ("paraphrase", "patient has T2DM", "SUPPORTED"),
    ("paraphrase", "patient has high blood pressure", "SUPPORTED"),
    ("paraphrase", "patient is on metformin", "SUPPORTED"),
]

if __name__ == "__main__":
    v = GraphVerifier()
    per, ok = Counter(), Counter()
    for i, (cat, text, gold) in enumerate(PROBES):
        got = v.verify(Assertion(id=str(i), text=text), EVIDENCE).state.value
        per[cat] += 1
        ok[cat] += got == gold
        if got != gold:
            print(f"MISMATCH [{cat}] {text!r}: expected {gold}, got {got}")
    for cat in per:
        print(f"{cat:18s} {ok[cat]}/{per[cat]}")
    print(f"TOTAL {sum(ok.values())}/{len(PROBES)}")
