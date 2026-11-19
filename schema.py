"""Core data structures shared across the NS-RAG pipeline.

These dataclasses give every module (graph construction, retrieval,
generation, verification, metrics) a single, unambiguous contract to
program against. Keeping them in one place also makes it easy to
serialize intermediate results to JSON for logging/reproducibility, as
required by the paper's protocol (Section 5.6, "Logged Outputs").
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Iterator, Optional


# --------------------------------------------------------------------------- #
# Patient graph primitives (GraphSynth, Section 3.2 of the paper)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Node:
    """A clinical entity or event node in the patient data graph DG_p."""
    id: str
    label: str            # e.g. "Condition", "Medication", "LabEvent", "Event"
    text: str              # surface form / canonical name
    timestamp: Optional[str] = None   # ISO8601 if temporally anchored
    attrs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    """A semantic or temporal relation between two nodes."""
    subj: str              # Node.id
    pred: str              # relation label, e.g. "treats", "precedes", "has_value"
    obj: str                # Node.id
    attrs: dict = field(default_factory=dict)


@dataclass
class PatientGraph:
    """The patient-centric data graph DG_p (before summarization)."""
    patient_id: str
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def n_nodes(self) -> int:
        return len(self.nodes)

    def n_edges(self) -> int:
        return len(self.edges)


@dataclass
class Triple:
    """An RDF-style (subject, predicate, object) triple, used as the
    common currency for graph-grounded context, retrieval, and
    verification (this is G^RDF_{p,T} / G*_{p,T} in the paper)."""
    subj: str
    pred: str
    obj: str
    provenance: Optional[str] = None   # source node/edge id for traceability

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subj, self.pred, self.obj)

    def as_text(self) -> str:
        return f"{self.subj} {self.pred} {self.obj}"


# --------------------------------------------------------------------------- #
# Assertions and verification (Section 3.4 / 5.8)
# --------------------------------------------------------------------------- #

class VerificationState(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"


@dataclass
class Assertion:
    """An atomic factual claim extracted from a generated (or reference)
    text, e.g. "patient received metformin on 2024-01-03"."""
    id: str
    text: str
    triple: Optional[Triple] = None     # normalized form, if extractable
    state: Optional[VerificationState] = None
    confidence: Optional[float] = None
    witness_subgraph: Optional[list[Triple]] = None   # G_exp for this assertion
    minimality_ratio: Optional[float] = None           # MR = |G_min| / |G_exp|, set after explanation extraction


@dataclass
class GenerationOutput:
    """Everything produced for one (patient, task) instance by one method.

    This is the unit that gets logged to disk (Section 5.6, "Logged
    Outputs") and consumed by every metric in `nsrag.metrics`.
    """
    method: str
    patient_id: str
    task: str                      # "qa" or "summarization"
    question: Optional[str]
    answer: str
    retrieved_context: list[str]
    context_tokens: int
    retrieval_time_s: float
    generation_time_s: float
    assertions: list[Assertion] = field(default_factory=list)
    minimal_explanation_graphs: dict[str, list[Triple]] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Small utilities used across modules
# --------------------------------------------------------------------------- #

@contextmanager
def timer() -> Iterator[Callable[[], float]]:
    """Context manager returning an accessor for elapsed wall-clock time.

    Usage:
        with timer() as t:
            do_work()
        elapsed_seconds = t()
    """
    start = time.perf_counter()
    box = {"end": None}

    def _get() -> float:
        end = box["end"] if box["end"] is not None else time.perf_counter()
        return end - start

    try:
        yield _get
    finally:
        box["end"] = time.perf_counter()
