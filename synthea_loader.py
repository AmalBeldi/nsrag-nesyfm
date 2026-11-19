"""Synthea synthetic patient data loader.

Synthea (https://github.com/synthetichealth/synthea) generates fully
synthetic patient records -- no real people, no DUA, no credentialing.
This is NOT a substitute for MIMIC-III/i2b2 in the paper's reported
results (Section 5.2 names those datasets specifically), but it lets
the entire pipeline run end-to-end today, with a real generation
backbone, producing real (not fabricated) outputs -- useful as a
wiring/scale validation pilot while MIMIC-III/i2b2 credentialing is
pending, and reportable as such (a synthetic-data robustness check) if
you choose to include it.

Key methodological property that makes Synthea uniquely convenient for
this purpose: because every row in conditions.csv/medications.csv IS
the ground truth (there is no annotation process, the simulator wrote
it), reference assertions for factuality scoring can be constructed
directly from the CSV rows with NO manual annotation step -- something
that is NOT true for MIMIC-III/i2b2, where reference assertions must
come from independent human annotation (Section 5.7).

Expected input: the `csv/` directory produced by running Synthea with
`exporter.csv.export = true` (patients.csv, conditions.csv,
medications.csv, observations.csv).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from nsrag.graph.graphsynth import ClinicalNote, StructuredEvent
from nsrag.pipeline.run_qa import QAInstance


@dataclass
class SyntheaPatient:
    patient_id: str
    first_name: str
    last_name: str
    conditions: list[dict]
    medications: list[dict]
    observations: list[dict]


class SyntheaLoader:
    def __init__(self, csv_dir: str | Path):
        self.dir = Path(csv_dir)

    def _read_csv(self, filename: str) -> list[dict]:
        path = self.dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run Synthea with exporter.csv.export = true "
                "and point `csv_dir` at the resulting output/csv/ directory."
            )
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Synthea's CSV header casing has varied across versions
        # (e.g. "PATIENT" vs "Patient"); normalize to uppercase keys
        # once here so the rest of this module can rely on a single
        # convention regardless of which version generated the file.
        return [{k.upper(): v for k, v in row.items()} for row in rows]

    def load_patients(self, limit: int | None = None) -> list[SyntheaPatient]:
        patients_rows = self._read_csv("patients.csv")
        conditions = self._read_csv("conditions.csv")
        medications = self._read_csv("medications.csv")
        observations = self._read_csv("observations.csv")

        def by_patient(rows: list[dict]) -> dict[str, list[dict]]:
            out: dict[str, list[dict]] = {}
            for row in rows:
                out.setdefault(row["PATIENT"], []).append(row)
            return out

        cond_by_patient = by_patient(conditions)
        med_by_patient = by_patient(medications)
        obs_by_patient = by_patient(observations)

        patients = []
        for row in patients_rows:
            pid = row["ID"]
            patients.append(SyntheaPatient(
                patient_id=pid,
                first_name=row.get("FIRST", ""), last_name=row.get("LAST", ""),
                conditions=cond_by_patient.get(pid, []),
                medications=med_by_patient.get(pid, []),
                observations=obs_by_patient.get(pid, []),
            ))
            if limit and len(patients) >= limit:
                break
        return patients

    @staticmethod
    def to_structured_events(patient: SyntheaPatient) -> list[StructuredEvent]:
        events = []
        for c in patient.conditions:
            events.append(StructuredEvent(
                patient_id=patient.patient_id, event_type="diagnosis",
                label=c["DESCRIPTION"], timestamp=c.get("START"),
            ))
        for m in patient.medications:
            events.append(StructuredEvent(
                patient_id=patient.patient_id, event_type="medication",
                label=m["DESCRIPTION"], timestamp=m.get("START"),
            ))
        for o in patient.observations:
            events.append(StructuredEvent(
                patient_id=patient.patient_id, event_type="lab",
                label=o["DESCRIPTION"], value=o.get("VALUE"), unit=o.get("UNITS"),
                timestamp=o.get("DATE"),
            ))
        return events

    @staticmethod
    def to_synthetic_note(patient: SyntheaPatient) -> ClinicalNote:
        """Builds a short templated clinical-note-style text from the
        structured ground truth. This is deliberately simple (not a
        substitute for real discharge-summary prose) -- it exists to
        give the entity/relation extractor free text to operate on
        without depending on Synthea's separate Java text exporter."""
        lines = [f"Patient {patient.first_name} {patient.last_name}."]
        for c in patient.conditions[:10]:
            lines.append(f"Diagnosed with {c['DESCRIPTION'].lower()}.")
        for m in patient.medications[:10]:
            lines.append(f"Started on {m['DESCRIPTION'].lower()}.")
        for o in patient.observations[:10]:
            if o.get("VALUE"):
                lines.append(f"{o['DESCRIPTION']} was {o['VALUE']} {o.get('UNITS', '')}.".strip())
        text = " ".join(lines)
        timestamp = (patient.conditions[0]["START"] if patient.conditions
                     else (patient.medications[0]["START"] if patient.medications else None))
        return ClinicalNote(patient_id=patient.patient_id, note_id="synthetic_note",
                             text=text, timestamp=timestamp)

    @staticmethod
    def to_reference_assertions(patient: SyntheaPatient) -> list[str]:
        """Ground-truth assertions derived directly from the CSV rows
        -- valid because Synthea's structured tables ARE the ground
        truth for a synthetic patient (see module docstring)."""
        assertions = []
        for c in patient.conditions:
            assertions.append(f"patient has {c['DESCRIPTION'].lower()}")
        for m in patient.medications:
            assertions.append(f"patient takes {m['DESCRIPTION'].lower()}")
        return assertions

    @staticmethod
    def make_question(patient: SyntheaPatient) -> str:
        """A simple templated question. This is a stand-in for real
        clinical questions (which the paper's protocol does not fix,
        see nsrag/scripts/common.py) -- adequate for a wiring/scale
        pilot, not for reporting as the paper's QA protocol."""
        if patient.conditions:
            condition = patient.conditions[0]["DESCRIPTION"]
            return f"What is the treatment plan for the patient's {condition.lower()}?"
        return "Summarize this patient's current clinical status."

    def to_qa_instances(self, limit: int | None = None) -> list[QAInstance]:
        instances = []
        for patient in self.load_patients(limit=limit):
            if not patient.conditions and not patient.medications:
                continue  # skip patients with no clinical content to ground a question in
            instances.append(QAInstance(
                patient_id=patient.patient_id,
                notes=[self.to_synthetic_note(patient)],
                structured_events=self.to_structured_events(patient),
                question=self.make_question(patient),
                reference_assertions=self.to_reference_assertions(patient),
            ))
        return instances
