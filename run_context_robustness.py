"""Context-length robustness runner (Figure 5, Section 5.11).

Sweeps the maximum context budget B over {2^7, ..., 2^11} tokens for
each method and records the resulting Hallucination Rate (== CR),
reproducing the exact experiment behind Figure 5.
"""
from __future__ import annotations

from dataclasses import dataclass

from nsrag.generation.generator import Generator, build_prompt
from nsrag.graph.dgsumm import DGsumm
from nsrag.graph.graphsynth import GraphSynth
from nsrag.graph.rdf_transform import HERSE
from nsrag.metrics.factuality import verification_partition
from nsrag.pipeline.run_qa import QAInstance
from nsrag.retrieval.graph_retriever import GraphRetriever, RetrievalConfig
from nsrag.retrieval.text_retriever import TextRetrievalConfig, TextRetriever
from nsrag.verification.assertion_extraction import extract_assertions
from nsrag.verification.verifier import GraphVerifier

CONTEXT_BUDGETS = [2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11]  # matches Figure 5's x-axis


@dataclass
class RobustnessPoint:
    method: str
    context_budget: int
    hallucination_rate: float


def _graph_context_at_budget(instance: QAInstance, budget: int) -> list:
    graph = GraphSynth().build(instance.patient_id, instance.notes, instance.structured_events)
    summary = DGsumm().summarize(graph, instance.question)
    triples = HERSE().to_triples(summary)
    retriever = GraphRetriever(RetrievalConfig(k=25, max_context_tokens=budget))
    return retriever.retrieve(triples, instance.question)


def _text_context_at_budget(instance: QAInstance, budget: int) -> list[str]:
    retriever = TextRetriever(TextRetrievalConfig(top_k=25, max_context_tokens=budget))
    return retriever.retrieve([n.text for n in instance.notes], instance.question)


def run_context_robustness_experiment(
    generator: Generator,
    instances: list[QAInstance],
    budgets: list[int] = CONTEXT_BUDGETS,
) -> list[RobustnessPoint]:
    verifier = GraphVerifier()
    points: list[RobustnessPoint] = []

    for budget in budgets:
        # Textual RAG: unstructured evidence, no graph to verify
        # against here -- included in Figure 5 for visual comparison
        # only; its HR in the paper's protocol is computed via the
        # same independent evaluator used in Table 5, not re-derived
        # from this sweep.
        for method_name, context_fn, linearize, verifiable in (
            ("Graph Linearization", _graph_context_at_budget,
             GraphRetriever.to_linearized_context, True),
            ("NS-RAG", _graph_context_at_budget,
             lambda ctx: "\n".join(GraphRetriever.to_structured_context(ctx)), True),
        ):
            all_assertions = []
            for inst in instances:
                ctx = context_fn(inst, budget)
                evidence_text = linearize(ctx)
                prompt = build_prompt(evidence=evidence_text, question=inst.question)
                result = generator.generate(prompt)
                assertions = extract_assertions(result.text)
                if verifiable:
                    assertions = verifier.verify_all(assertions, ctx)
                    all_assertions.extend(assertions)

            if all_assertions:
                hr = verification_partition(all_assertions).cr
                points.append(RobustnessPoint(method_name, budget, hr))

    return points
