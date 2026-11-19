"""End-to-end runner for the Clinical QA experiment (Table 5, RQ1/RQ2/RQ4).

Usage (see README.md for the full walkthrough):

    from nsrag.pipeline.run_qa import run_qa_experiment, QAInstance

    instances = [QAInstance(patient_id=..., notes=..., structured_events=...,
                             question=..., reference_assertions=[...]), ...]
    results = run_qa_experiment(methods, instances)
    print(results)   # -> feed straight into utils.results_to_latex

This module intentionally has ZERO awareness of *which* generator or
extractor backend is plugged into each `Method` -- swap `MockGenerator`
for `HFGenerator` (Section 5.6) to go from a wiring smoke test to a
real experimental run without touching this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from nsrag.baselines.common import Method
from nsrag.graph.graphsynth import ClinicalNote, StructuredEvent
from nsrag.metrics.efficiency import StageTimer
from nsrag.metrics.factuality import factuality_score, verification_partition
from nsrag.schema import GenerationOutput
from nsrag.verification.assertion_extraction import extract_assertions


@dataclass
class QAInstance:
    patient_id: str
    notes: list[ClinicalNote]
    structured_events: list[StructuredEvent]
    question: str
    reference_assertions: list[str] = field(default_factory=list)


@dataclass
class MethodResult:
    method: str
    fact_precision: float
    fact_recall: float
    fact_f1: float
    cr: float | None
    ur: float | None
    ec: float | None
    n_instances: int
    outputs: list[GenerationOutput] = field(default_factory=list)


def run_qa_experiment(
    methods: list[Method],
    instances: list[QAInstance],
    match_threshold: float = 0.5,
) -> dict[str, MethodResult]:
    results: dict[str, MethodResult] = {}
    timer = StageTimer()

    for method in methods:
        outputs: list[GenerationOutput] = []
        fact_scores = []
        has_verification = False
        all_assertions_for_partition = []

        for inst in instances:
            output = method.run(
                inst.patient_id, inst.notes, inst.structured_events,
                inst.question, task="qa",
            )
            timer.record(f"{method.name}::retrieval", output.retrieval_time_s)
            timer.record(f"{method.name}::generation", output.generation_time_s)
            outputs.append(output)

            # Factuality is evaluated uniformly across ALL methods by
            # re-extracting assertions from the generated answer and
            # comparing to the independently constructed reference
            # assertions (Section 5.7/5.10), regardless of whether the
            # method itself produces assertion-level output.
            gen_assertions = output.assertions or extract_assertions(output.answer)
            if inst.reference_assertions:
                fact_scores.append(
                    factuality_score(gen_assertions, inst.reference_assertions, match_threshold)
                )

            if output.assertions and any(a.state is not None for a in output.assertions):
                has_verification = True
                all_assertions_for_partition.extend(output.assertions)

        n = max(len(fact_scores), 1)
        precision = sum(s.precision for s in fact_scores) / n if fact_scores else 0.0
        recall = sum(s.recall for s in fact_scores) / n if fact_scores else 0.0
        f1 = sum(s.f1 for s in fact_scores) / n if fact_scores else 0.0

        cr = ur = ec = None
        if has_verification:
            part = verification_partition(all_assertions_for_partition)
            cr, ur, ec = part.cr, part.ur, part.ec

        results[method.name] = MethodResult(
            method=method.name, fact_precision=round(precision, 4),
            fact_recall=round(recall, 4), fact_f1=round(f1, 4),
            cr=cr, ur=ur, ec=ec, n_instances=len(instances), outputs=outputs,
        )

    results["_efficiency"] = timer.summary()  # type: ignore[assignment]
    return results
