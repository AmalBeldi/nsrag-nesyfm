"""GraphSynth: patient-centric EHR graph construction (paper Section 3.2).

Builds the longitudinal patient data graph DG_p from heterogeneous EHR
inputs: discharge/clinical notes, structured lab events, medication
orders, and diagnosis codes. Nodes are clinical entities/events;
edges encode semantic relations (extracted from text) and temporal
relations (derived from timestamps already present in structured
tables).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Optional

from nsrag.graph.extraction import EntityExtractor, RelationExtractor, RuleBasedExtractor
from nsrag.schema import Edge, Node, PatientGraph


def _stable_id(*parts: str) -> str:
    """Deterministic node id so re-running graph construction on the
    same inputs yields identical node identifiers (needed for the
    reproducibility guarantees described in Section 5.18)."""
    h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"n_{h}"


@dataclass
class ClinicalNote:
    patient_id: str
    note_id: str
    text: str
    timestamp: Optional[str] = None
    note_type: str = "discharge_summary"


@dataclass
class StructuredEvent:
    """A row from a structured MIMIC-III table (LABEVENTS,
    PRESCRIPTIONS, DIAGNOSES_ICD, ...) already normalized to a common
    shape by `nsrag.data.mimic_loader`."""
    patient_id: str
    event_type: str        # "lab", "medication", "diagnosis", "procedure"
    label: str              # e.g. "Glucose", "Metformin", "Type 2 Diabetes"
    value: Optional[str] = None
    unit: Optional[str] = None
    timestamp: Optional[str] = None


class GraphSynth:
    """Constructs a `PatientGraph` from clinical notes and structured
    events for a single patient."""

    def __init__(
        self,
        entity_extractor: Optional[EntityExtractor] = None,
        relation_extractor: Optional[RelationExtractor] = None,
    ):
        default = RuleBasedExtractor()
        self.entity_extractor = entity_extractor or default
        self.relation_extractor = relation_extractor or default

    def build(
        self,
        patient_id: str,
        notes: Iterable[ClinicalNote] = (),
        structured_events: Iterable[StructuredEvent] = (),
    ) -> PatientGraph:
        graph = PatientGraph(patient_id=patient_id)

        # 1) Structured events -> nodes directly (already discrete facts).
        prev_event_node: Optional[str] = None
        structured_events = sorted(
            structured_events, key=lambda e: (e.timestamp or "")
        )
        for ev in structured_events:
            node_id = _stable_id(patient_id, ev.event_type, ev.label, ev.timestamp or "")
            graph.add_node(Node(
                id=node_id, label=ev.event_type.capitalize(), text=ev.label,
                timestamp=ev.timestamp,
                attrs={"value": ev.value, "unit": ev.unit},
            ))
            # Temporal chaining: precedes edges give the retriever and
            # verifier access to simple temporal ordering without a
            # full temporal-reasoning engine.
            if prev_event_node is not None:
                graph.add_edge(Edge(subj=prev_event_node, pred="precedes", obj=node_id))
            prev_event_node = node_id

        # 2) Free-text notes -> entities + relations via the pluggable extractor.
        for note in notes:
            entities = self.entity_extractor.extract_entities(note.text)
            entity_nodes: dict[int, str] = {}
            for i, ent in enumerate(entities):
                node_id = _stable_id(patient_id, note.note_id, str(ent.start), ent.text)
                graph.add_node(Node(
                    id=node_id, label=ent.label, text=ent.text,
                    timestamp=note.timestamp,
                    attrs={"source_note": note.note_id},
                ))
                entity_nodes[i] = node_id

            if isinstance(self.relation_extractor, RelationExtractor) or hasattr(
                self.relation_extractor, "extract_relations"
            ):
                relations = self.relation_extractor.extract_relations(note.text, entities)
                for rel in relations:
                    subj_idx = entities.index(rel.subj)
                    obj_idx = entities.index(rel.obj)
                    graph.add_edge(Edge(
                        subj=entity_nodes[subj_idx], pred=rel.pred, obj=entity_nodes[obj_idx],
                        attrs={"source_note": note.note_id},
                    ))

        return graph
