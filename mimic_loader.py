"""MIMIC-III loader.

MIMIC-III requires completing CITI human-subjects training and signing
a data use agreement on PhysioNet before you can download it:
https://physionet.org/content/mimiciii/

This loader assumes the standard PhysioNet CSV export layout
(NOTEEVENTS.csv, LABEVENTS.csv, PRESCRIPTIONS.csv, DIAGNOSES_ICD.csv,
D_LABITEMS.csv, D_ICD_DIAGNOSES.csv) and normalizes rows into the
`ClinicalNote` / `StructuredEvent` shapes the rest of the pipeline
expects. No PHI is bundled with this repository -- point `mimic_dir`
at your own credentialed local copy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nsrag.graph.graphsynth import ClinicalNote, StructuredEvent


@dataclass
class MimicSplit:
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]


class MimicLoader:
    def __init__(self, mimic_dir: str | Path):
        self.dir = Path(mimic_dir)

    def _read(self, filename: str) -> pd.DataFrame:
        path = self.dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download MIMIC-III from PhysioNet "
                "(credentialed access required) and point `mimic_dir` at "
                "the directory containing the raw CSVs."
            )
        df = pd.read_csv(path, low_memory=False)
        # Column-name casing has varied across MIMIC-III distributions
        # (the credentialed release uses UPPERCASE headers, e.g.
        # SUBJECT_ID; the openly-accessible MIMIC-III Demo uses
        # lowercase headers, e.g. subject_id). Normalize to uppercase
        # once here so the rest of this class can rely on a single
        # convention regardless of which release produced the file.
        df.columns = [c.upper() for c in df.columns]
        return df

    def load_notes(self, patient_id: str, note_type: str = "Discharge summary") -> list[ClinicalNote]:
        df = self._read("NOTEEVENTS.csv")
        rows = df[(df["SUBJECT_ID"].astype(str) == str(patient_id)) & (df["CATEGORY"] == note_type)]
        return [
            ClinicalNote(
                patient_id=str(patient_id), note_id=str(row["ROW_ID"]),
                text=str(row["TEXT"]), timestamp=str(row.get("CHARTDATE")),
                note_type=note_type,
            )
            for _, row in rows.iterrows()
        ]

    def load_structured_events(self, patient_id: str) -> list[StructuredEvent]:
        events: list[StructuredEvent] = []

        try:
            labs = self._read("LABEVENTS.csv")
            items = self._read("D_LABITEMS.csv").set_index("ITEMID")["LABEL"]
            patient_labs = labs[labs["SUBJECT_ID"].astype(str) == str(patient_id)]
            for _, row in patient_labs.iterrows():
                label = items.get(row["ITEMID"], f"lab_{row['ITEMID']}")
                events.append(StructuredEvent(
                    patient_id=str(patient_id), event_type="lab", label=str(label),
                    value=str(row.get("VALUE")), unit=str(row.get("VALUEUOM")),
                    timestamp=str(row.get("CHARTTIME")),
                ))
        except FileNotFoundError:
            pass

        try:
            rx = self._read("PRESCRIPTIONS.csv")
            patient_rx = rx[rx["SUBJECT_ID"].astype(str) == str(patient_id)]
            for _, row in patient_rx.iterrows():
                events.append(StructuredEvent(
                    patient_id=str(patient_id), event_type="medication",
                    label=str(row.get("DRUG")), value=str(row.get("DOSE_VAL_RX")),
                    unit=str(row.get("DOSE_UNIT_RX")), timestamp=str(row.get("STARTDATE")),
                ))
        except FileNotFoundError:
            pass

        try:
            dx = self._read("DIAGNOSES_ICD.csv")
            dx_labels = self._read("D_ICD_DIAGNOSES.csv").set_index("ICD9_CODE")["SHORT_TITLE"]
            patient_dx = dx[dx["SUBJECT_ID"].astype(str) == str(patient_id)]
            for _, row in patient_dx.iterrows():
                label = dx_labels.get(row["ICD9_CODE"], f"icd9_{row['ICD9_CODE']}")
                events.append(StructuredEvent(
                    patient_id=str(patient_id), event_type="diagnosis", label=str(label),
                ))
        except FileNotFoundError:
            pass

        return events

    def make_split(self, seed: int = 42, train_frac: float = 0.7, val_frac: float = 0.1) -> MimicSplit:
        """70/10/20 split as specified in Section 5.2."""
        df = self._read("NOTEEVENTS.csv")
        ids = sorted(df["SUBJECT_ID"].astype(str).unique().tolist())
        import random
        rng = random.Random(seed)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        return MimicSplit(
            train_ids=ids[:n_train],
            val_ids=ids[n_train:n_train + n_val],
            test_ids=ids[n_train + n_val:],
        )
