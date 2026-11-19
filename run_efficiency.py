"""Stage-level efficiency runner (Table 9, Section 5.17).

Instruments every stage of the NS-RAG pipeline individually --
graph construction, graph summarization, RDF transformation, context
construction, LLM generation, verification, explanation -- rather than
just retrieval/generation as in `run_qa.py`'s coarser timing, matching
Table 9's row breakdown exactly.
"""
from __future__ import annotations

from nsrag.generation.generator import Generator, build_prompt
from nsrag.graph.dgsumm import DGsumm
from nsrag.graph.graphsynth import GraphSynth
from nsrag.graph.rdf_transform import HERSE
from nsrag.metrics.efficiency import StageTimer
from nsrag.pipeline.run_qa import QAInstance
from nsrag.retrieval.graph_retriever import GraphRetriever
from nsrag.schema import timer
from nsrag.verification.assertion_extraction import extract_assertions
from nsrag.verification.explain import extract_minimal_explanation
from nsrag.verification.verifier import GraphVerifier


def run_efficiency_experiment(
    generator: Generator,
    instances: list[QAInstance],
) -> dict:
    stage_timer = StageTimer()
    graphsynth, dgsumm, herse, retriever, verifier = (
        GraphSynth(), DGsumm(), HERSE(), GraphRetriever(), GraphVerifier(),
    )

    for inst in instances:
        with timer() as t:
            full_graph = graphsynth.build(inst.patient_id, inst.notes, inst.structured_events)
        stage_timer.record("Graph construction", t())

        with timer() as t:
            summary_graph = dgsumm.summarize(full_graph, inst.question)
        stage_timer.record("Graph summarization", t())

        with timer() as t:
            triples = herse.to_triples(summary_graph)
        stage_timer.record("RDF transformation", t())

        with timer() as t:
            selected = retriever.retrieve(triples, inst.question)
            evidence_text = "\n".join(GraphRetriever.to_structured_context(selected))
        stage_timer.record("Context construction", t())

        with timer() as t:
            prompt = build_prompt(evidence=evidence_text, question=inst.question)
            result = generator.generate(prompt)
        stage_timer.record("LLM generation", t())

        with timer() as t:
            assertions = extract_assertions(result.text)
            assertions = verifier.verify_all(assertions, selected)
        stage_timer.record("Verification", t())

        with timer() as t:
            for a in assertions:
                if a.witness_subgraph:
                    extract_minimal_explanation(a, a.witness_subgraph, verifier)
        stage_timer.record("Explanation", t())

    return stage_timer.summary()
