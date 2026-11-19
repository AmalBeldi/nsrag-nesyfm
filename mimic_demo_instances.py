"""Builds QAInstance objects from real MIMIC-III data via MimicLoader.

Works with either:
  - the MIMIC-III Clinical Database Demo (open access, no credentialing,
    100 real de-identified patients, https://physionet.org/content/mimiciii-demo/) --
    NOTEEVENTS.csv is present but empty (notes were stripped for the demo);
  - the full credentialed MIMIC-III database, where NOTEEVENTS.csv is
    populated with real discharge summaries.

In both cases, structured data (diagnoses, medications, labs) is real,
de-identified patient data -- not simulated. When free-text notes are
unavailable (the demo case), a templated note is synthesized from the
real structured facts, exactly as for the Synthea pilot
(nsrag/data/synthea_loader.py), so GraphSynth's entity/relation
extractor has text to operate on. This is a documented compromise, not
a claim that the note text itself is authentic clinical prose.

Because MIMIC-III (including the demo) provides no public question set
or independently-annotated reference assertions for the Clinical QA
task, both are constructed here from the real structured facts using
the same templating approach as the Synthea pilot (see
nsrag/data/synthea_loader.py for the rationale): reference assertions
are derived directly from the real DIAGNOSES_ICD / PRESCRIPTIONS rows
(these ARE real patient facts, even though the phrasing is templated),
which is a meaningfully stronger basis than the fully synthetic pilot.
"""
from __future__ import annotations

from nsrag.data.mimic_loader import MimicLoader
from nsrag.graph.graphsynth import ClinicalNote, StructuredEvent
from nsrag.pipeline.run_qa import QAInstance


def _to_synthetic_note(patient_id: str, events: list[StructuredEvent]) -> ClinicalNote:
    diagnoses = [e for e in events if e.event_type == "diagnosis"]
    medications = [e for e in events if e.event_type == "medication"]
    labs = [e for e in events if e.event_type == "lab"]

    lines = [f"Patient {patient_id}."]
    for d in diagnoses[:10]:
        lines.append(f"Diagnosed with {d.label.lower()}.")
    for m in medications[:10]:
        lines.append(f"Started on {m.label.lower()}.")
    for l in labs[:10]:
        if l.value:
            lines.append(f"{l.label} was {l.value} {l.unit or ''}.".strip())

    text = " ".join(lines) if len(lines) > 1 else f"Patient {patient_id}. No structured events on record."
    timestamp = (diagnoses[0].timestamp if diagnoses
                 else (medications[0].timestamp if medications else None))
    return ClinicalNote(patient_id=patient_id, note_id="synthesized_from_structured_data",
                         text=text, timestamp=timestamp)


def _make_question(events: list[StructuredEvent]) -> str:
    diagnoses = [e for e in events if e.event_type == "diagnosis"]
    if diagnoses:
        return f"What is the treatment plan for the patient's {diagnoses[0].label.lower()}?"
    return "Summarize this patient's current clinical status."


def _reference_assertions(events: list[StructuredEvent]) -> list[str]:
    assertions = []
    for e in events:
        if e.event_type == "diagnosis":
            assertions.append(f"patient has {e.label.lower()}")
        elif e.event_type == "medication":
            assertions.append(f"patient takes {e.label.lower()}")
    return assertions


def build_qa_instances_from_mimic(
    loader: MimicLoader,
    patient_ids: list[str],
    real_note_type: str | None = "Discharge summary",
) -> list[QAInstance]:
    """For each patient id, tries to load real notes first (works
    unmodified against the full credentialed MIMIC-III, where
    NOTEEVENTS.csv is populated); falls back to a templated note
    synthesized from real structured events when no real notes are
    found (the MIMIC-III Demo case, or any patient with no discharge
    summary on file).
    """
    instances = []
    for pid in patient_ids:
        events = loader.load_structured_events(pid)
        if not events:
            continue  # nothing to ground a question in

        notes: list[ClinicalNote] = []
        if real_note_type:
            try:
                notes = loader.load_notes(pid, note_type=real_note_type)
            except Exception:
                notes = []
        if not notes:
            notes = [_to_synthetic_note(pid, events)]

        instances.append(QAInstance(
            patient_id=pid,
            notes=notes,
            structured_events=events,
            question=_make_question(events),
            reference_assertions=_reference_assertions(events),
        ))
    return instances
