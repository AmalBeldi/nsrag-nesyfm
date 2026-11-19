"""Shared CLI plumbing: load `configs/default.yaml`, build the
generator backend, and load QA/summarization instances from a JSONL
file prepared by the user.

Why instances come from a JSONL file rather than being auto-generated
from MIMIC-III here: the paper's protocol does not specify a public
question-generation procedure for the Clinical QA task (Section 5.2
defines the *task*, not a fixed question set), so manufacturing one
inside this pipeline would silently bake in an unreported experimental
choice. Prepare your instance file explicitly instead -- see
`README.md` -> "Preparing instances" for the expected JSONL schema and
a script sketch for deriving it from MIMIC-III + your own QA source
(e.g. clinician-written questions, or a templated question generator
you report in the paper).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from nsrag.generation.generator import Generator, HFGenerator, MockGenerator
from nsrag.graph.graphsynth import ClinicalNote, StructuredEvent
from nsrag.pipeline.run_qa import QAInstance
from nsrag.pipeline.run_summarization import SummarizationInstance


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--instances", help="JSONL file of prepared instances (see README.md)")
    p.add_argument("--out", required=True, help="Where to write the JSON results")
    p.add_argument(
        "--mock", action="store_true",
        help="Use MockGenerator instead of the real HF backend (wiring/CI checks only; "
             "NEVER use --mock to produce numbers for the paper).",
    )
    return p


def build_generator(config: dict, mock: bool) -> Generator:
    if mock:
        return MockGenerator()
    gen_cfg = config["generation"]
    return HFGenerator(
        model_name=gen_cfg["backbone"],
        device=gen_cfg.get("device", "cuda"),
        temperature=gen_cfg.get("temperature", 0.2),
    )


def _parse_note(d: dict) -> ClinicalNote:
    return ClinicalNote(
        patient_id=d["patient_id"], note_id=d.get("note_id", "n0"),
        text=d["text"], timestamp=d.get("timestamp"),
        note_type=d.get("note_type", "discharge_summary"),
    )


def _parse_event(d: dict) -> StructuredEvent:
    return StructuredEvent(
        patient_id=d["patient_id"], event_type=d["event_type"], label=d["label"],
        value=d.get("value"), unit=d.get("unit"), timestamp=d.get("timestamp"),
    )


def load_qa_instances(path: str) -> list[QAInstance]:
    """Expected JSONL schema, one object per line:

        {
          "patient_id": "12345",
          "notes": [{"note_id": "n1", "text": "...", "timestamp": "2024-01-01"}],
          "structured_events": [{"event_type": "lab", "label": "Glucose",
                                  "value": "210", "timestamp": "2024-01-02"}],
          "question": "What is the glucose control plan for this patient?",
          "reference_assertions": ["patient has type 2 diabetes", "..."]
        }
    """
    instances = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        for note in d.get("notes", []):
            note.setdefault("patient_id", d["patient_id"])
        for ev in d.get("structured_events", []):
            ev.setdefault("patient_id", d["patient_id"])
        instances.append(QAInstance(
            patient_id=d["patient_id"],
            notes=[_parse_note(n) for n in d.get("notes", [])],
            structured_events=[_parse_event(e) for e in d.get("structured_events", [])],
            question=d["question"],
            reference_assertions=d.get("reference_assertions", []),
        ))
    return instances


def load_summarization_instances(path: str) -> list[SummarizationInstance]:
    """Same schema as `load_qa_instances`, but with `reference_summary`
    (required) instead of `question`, and `reference_assertions`
    (optional, for factuality scoring alongside ROUGE-L/BERTScore)."""
    instances = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        for note in d.get("notes", []):
            note.setdefault("patient_id", d["patient_id"])
        for ev in d.get("structured_events", []):
            ev.setdefault("patient_id", d["patient_id"])
        instances.append(SummarizationInstance(
            patient_id=d["patient_id"],
            notes=[_parse_note(n) for n in d.get("notes", [])],
            structured_events=[_parse_event(e) for e in d.get("structured_events", [])],
            reference_summary=d["reference_summary"],
            reference_assertions=d.get("reference_assertions", []),
        ))
    return instances


def write_json(obj, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"Wrote {out_path}")
