"""Perturbation test of the three-state verifier on REAL MIMIC-III Demo records.

No generator involved. For each patient, evidence = typed triples built from
the patient's own structured diagnoses/medications. Assertions are built with
gold labels *by construction*:
  supported        : own diagnosis/medication            -> SUPPORTED
  negated_present  : negation of an own fact             -> CONTRADICTED
  absent           : fact taken from ANOTHER patient and
                     not in this record                  -> UNKNOWN
  negated_absent   : negation of an absent fact          -> UNKNOWN
Caveat: gold labels follow from the record (circular by design); this tests
verifier logic and schema handling at scale, not clinical truth. Lab values
are excluded: they are time-indexed, so a different value is not a conflict.
Usage: PYTHONPATH=src python scripts/verification_perturbation_eval.py --mimic-dir DIR
"""
import argparse, random, json
from collections import Counter, defaultdict
from nsrag.data.mimic_loader import MimicLoader
from nsrag.schema import Assertion, Triple
from nsrag.verification.verifier import GraphVerifier


def clean(lbl):
    s = str(lbl).strip().lower()
    return None if (not s or s == "nan" or s.startswith("icd9_")) else " ".join(s.split())


def patient_facts(loader, pid):
    dx, rx = set(), set()
    for e in loader.load_structured_events(pid):
        l = clean(e.label)
        if l is None:
            continue
        if e.event_type == "diagnosis":
            dx.add(l)
        elif e.event_type == "medication":
            rx.add(l)
    return dx, rx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-dir", required=True)
    ap.add_argument("--per-kind", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="perturbation_results.json")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    loader = MimicLoader(a.mimic_dir)
    pids = loader._read("PATIENTS.csv")["SUBJECT_ID"].astype(str).unique().tolist()
    facts = {p: patient_facts(loader, p) for p in pids}
    facts = {p: f for p, f in facts.items() if f[0] or f[1]}
    all_dx = sorted({x for d, _ in facts.values() for x in d})
    all_rx = sorted({x for _, r in facts.values() for x in r})
    v = GraphVerifier()
    rows = []
    for pid, (dx, rx) in facts.items():
        ev = [Triple("Patient", "has_condition", x) for x in sorted(dx)] + \
             [Triple("Patient", "takes_medication", x) for x in sorted(rx)]
        own_dx = rng.sample(sorted(dx), min(a.per_kind, len(dx)))
        own_rx = rng.sample(sorted(rx), min(a.per_kind, len(rx)))
        abs_dx = rng.sample([x for x in all_dx if x not in dx], min(a.per_kind, len(all_dx) - len(dx)))
        abs_rx = rng.sample([x for x in all_rx if x not in rx], min(a.per_kind, len(all_rx) - len(rx)))
        cases = []
        cases += [("supported", f"patient has {x}", "SUPPORTED") for x in own_dx]
        cases += [("supported", f"patient takes {x}", "SUPPORTED") for x in own_rx]
        cases += [("negated_present", f"patient has no {x}", "CONTRADICTED") for x in own_dx]
        cases += [("negated_present", f"patient does not take {x}", "CONTRADICTED") for x in own_rx]
        cases += [("absent", f"patient has {x}", "UNKNOWN") for x in abs_dx]
        cases += [("absent", f"patient takes {x}", "UNKNOWN") for x in abs_rx]
        cases += [("negated_absent", f"patient has no {x}", "UNKNOWN") for x in abs_dx]
        cases += [("negated_absent", f"patient does not take {x}", "UNKNOWN") for x in abs_rx]
        for i, (cat, text, gold) in enumerate(cases):
            got = v.verify(Assertion(id=f"{pid}_{i}", text=text), ev).state.value
            rows.append((cat, gold, got, text))
    by = defaultdict(lambda: [0, 0])
    conf = Counter()
    for cat, gold, got, _ in rows:
        by[cat][1] += 1
        by[cat][0] += gold == got
        conf[(gold, got)] += 1
    print(f"patients={len(facts)} assertions={len(rows)}")
    for cat, (ok, n) in by.items():
        print(f"{cat:16s} {ok}/{n} = {ok/n:.3f}")
    tot = sum(o for o, _ in by.values())
    print(f"overall          {tot}/{len(rows)} = {tot/len(rows):.3f}")
    print("confusion (gold,pred):count")
    for k, c in sorted(conf.items()):
        print(" ", k, c)
    errs = [r for r in rows if r[1] != r[2]][:15]
    print("examples of errors:")
    for r in errs:
        print("  ", r)
    json.dump({"by_category": {k: v_ for k, v_ in by.items()},
               "confusion": {f"{g}->{p}": c for (g, p), c in conf.items()}},
              open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
