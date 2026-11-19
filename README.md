# NS-RAG: graph-grounded three-state verification for clinical QA

Open pipeline accompanying a short position paper on **graph-grounded,
open-world, three-state verification** (SUPPORTED / CONTRADICTED /
UNKNOWN) of generated clinical answers. Status: **work in progress**.
Read "Status and limitations" before using any number from this code.

## Pipeline

Patient graph (GraphSynth) -> task-oriented summarization (DGsumm) ->
RDF triples -> graph-grounded retrieval (k=25, 2048-token budget) ->
generation (Flan-T5-Large, temperature 0.2) -> assertion extraction ->
three-state verification with witness triples -> minimal explanation
subgraph (greedy) and Minimality Ratio.

Verification semantics (see `src/nsrag/verification/verifier.py`):
SUPPORTED if a triple matches (Jaccard overlap >= 0.5), polarity agrees
and numbers agree; CONTRADICTED on negation mismatch, or on an
incompatible value of a *functional* predicate (`lab_*`) naming the same
analyte; UNKNOWN otherwise (open-world). Matching is lexical, not NLI.

## What is reproducible here

| Item | How |
|---|---|
| Unit tests (46) | `python -m unittest discover -s tests -v` |
| 22-probe suite on the verifier semantics | `PYTHONPATH=src python scripts/verification_probes.py` (expects 19/22; the 3 misses are paraphrases) |
| Perturbation test on real MIMIC-III Demo records | `PYTHONPATH=src python scripts/verification_perturbation_eval.py --mimic-dir <demo dir>` |
| Synthea pilot (synthetic) | `notebooks/synthetic_pilot_colab.ipynb` |
| MIMIC-III Demo pilot | `notebooks/mimic_demo_colab.ipynb` |

Both notebooks contain a `git clone` of a placeholder
(`https://github.com/ANONYMOUS/REPO.git`); replace it with this
repository's address, or upload the code manually.

## Data

- **MIMIC-III Clinical Database Demo** (100 de-identified patients,
  Open Data Commons ODbL, no credentialing):
  <https://physionet.org/content/mimiciii-demo/1.4/>. Download the CSVs
  (`PATIENTS`, `ADMISSIONS`, `DIAGNOSES_ICD`, `D_ICD_DIAGNOSES`,
  `PRESCRIPTIONS`, `LABEVENTS`, `D_LABITEMS`, `NOTEEVENTS`). Free-text
  notes are empty in the demo; notes are synthesised from structured facts.
  Headers are lowercase in the demo; the loader normalises them.
- **Synthea** synthetic patients (generated locally).
- Full MIMIC-III and i2b2 loaders exist but were **never run**: the
  authors have no credentialed access. No data is bundled.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # spacy may fail on some platforms; not needed for the tests
```

## Status and limitations (please read)

- Reported results: verification semantics, the probe suite, a
  perturbation test on 100 real records (gold labels **by construction**,
  typed evidence built from structured data), and two pilots that
  validate **execution only**.
- In the pilots the verifier is largely inactive: generated answers are
  one-word, assertion extraction is mostly empty, and the pipeline's
  generic RDF export (`rdf:type`, `hasText`, ...) does not use the typed
  predicates (`has_condition`, `takes_medication`, `lab_*`) assumed by the
  verifier. Pilot Fact-F1 values are near zero for every method; do not
  interpret them as quality measurements. Questions are templated and
  references derive from the same structured data as the graph (circular).
- The probes were written after observing an earlier closed-world defect
  and the verifier was tuned on them: 19/22 is a regression score, not a
  held-out estimate.
- Perturbation test (v1, 3,930 assertions, 100 patients): macro-F1
  0.885; negated present facts recovered at 0.62; 30 of 2,000 UNKNOWN-gold
  assertions mislabelled (18 shared an active ingredient; 12 clear errors).
- Lab values are time-indexed; the verifier does not model time.
- Not done: GraphRAG/LightRAG runs (`baselines/graph_rag_family.py` is
  untested), human-annotated assertion states, explanation-faithfulness
  evaluation, a repaired ablation protocol (an earlier ablation changed
  Fact-F1 when verification was removed, which should not happen).
- `MockGenerator` is for wiring tests only; never use it for reported numbers.

## Layout

```
src/nsrag/{graph,retrieval,generation,verification,baselines,metrics,data,pipeline,scripts,utils}
scripts/   verification_probes.py, verification_perturbation_eval.py, run_all.sh
notebooks/ synthetic_pilot_colab.ipynb, mimic_demo_colab.ipynb
tests/     unittest suite (46 tests)
configs/default.yaml
sample_data/  tiny toy examples for smoke tests
```
