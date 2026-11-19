"""Ablation study runner (Table 8 / Figure 6, RQ5).

Produces the four ablated NS-RAG configurations described in Section
5.14 by toggling one component at a time, reusing the exact same
GraphSynth/DGsumm/HERSE/Retriever/Verifier building blocks as B8 so
that the *only* difference between rows is the ablated component
(no confound from re-implementing pieces differently).
"""
from __future__ import annotations

from dataclasses import dataclass

from nsrag.baselines.ns_rag import NSRAG, _NSRAGBase
from nsrag.generation.generator import Generator
from nsrag.graph.dgsumm import DGsumm, DGsummConfig
from nsrag.graph.graphsynth import GraphSynth
from nsrag.graph.rdf_transform import HERSE
from nsrag.pipeline.run_qa import QAInstance, run_qa_experiment
from nsrag.retrieval.graph_retriever import GraphRetriever, RetrievalConfig
from nsrag.schema import GenerationOutput, Triple, timer
from nsrag.verification.assertion_extraction import extract_assertions
from nsrag.verification.explain import extract_minimal_explanation, minimality_ratio
from nsrag.verification.verifier import GraphVerifier


class _NoOpDGsumm(DGsumm):
    """'-- graph summarization': skip reduction, pass the full graph
    through unchanged (still goes through HERSE/RDF transform)."""

    def summarize(self, graph, task_query):
        return graph


class _NoOpVerifier(GraphVerifier):
    """'-- symbolic verification': every assertion is left UNKNOWN
    (verification is skipped, exactly as B7 does) rather than actually
    matched against evidence."""

    def verify(self, assertion, evidence):
        assertion.state = None
        assertion.confidence = None
        assertion.witness_subgraph = None
        return assertion


class _PassthroughRetriever(GraphRetriever):
    """'-- graph-grounded retrieval': instead of scoring/selecting
    triples by relevance to the question, take an arbitrary (relevance
    -blind) subset up to the same budget, to isolate the *value of
    relevance-based selection* rather than of having any graph context
    at all."""

    def retrieve(self, triples: list[Triple], question: str) -> list[Triple]:
        return triples[: self.config.k]


class _NoExplanationNSRAG(NSRAG):
    """'-- minimal explanations': keep verification, but report the
    full witness set G_exp as the explanation instead of minimizing it
    to G_min (MR = 1.0 by construction for every SUPPORTED assertion)."""

    def run(self, patient_id, notes, structured_events, question, task) -> GenerationOutput:
        selected, retrieval_time = self._retrieve(patient_id, notes, structured_events, question)
        evidence_text = "\n".join(GraphRetriever.to_structured_context(selected))
        result, _ = self._generate(evidence_text=evidence_text, question=question)

        assertions = extract_assertions(result.text, id_prefix=f"{patient_id}_a")
        assertions = self.verifier.verify_all(assertions, selected)
        for a in assertions:
            if a.witness_subgraph:
                a.minimality_ratio = 1.0  # G_min == G_exp, no minimization performed

        return GenerationOutput(
            method=self.name, patient_id=patient_id, task=task, question=question,
            answer=result.text, retrieved_context=[t.as_text() for t in selected],
            context_tokens=sum(len(t.as_text().split()) for t in selected),
            retrieval_time_s=retrieval_time, generation_time_s=result.generation_time_s,
            assertions=assertions,
        )


def build_ablation_configurations(generator: Generator) -> dict[str, _NSRAGBase]:
    """Returns the five configurations of Table 8, keyed by row label."""
    return {
        "Full model (NS-RAG)": NSRAG(generator),
        "-- symbolic verification": NSRAG(generator, verifier=_NoOpVerifier()),
        "-- graph summarization": NSRAG(generator, dgsumm=_NoOpDGsumm()),
        "-- graph-grounded retrieval": NSRAG(generator, retriever=_PassthroughRetriever(RetrievalConfig())),
        "-- minimal explanations": _NoExplanationNSRAG(generator),
    }


def run_ablation_experiment(generator: Generator, instances: list[QAInstance]) -> dict:
    """Runs each ablated configuration independently (rather than as one
    batch through `run_qa_experiment`) because every configuration is a
    variant of the *same* `NSRAG` class and therefore shares the method
    name "B8 - NS-RAG" -- batching them would collide on that name as a
    dict key. Each is relabeled with its Table 8 row label instead.
    """
    configs = build_ablation_configurations(generator)
    relabeled = {}
    efficiency = {}
    for label, method in configs.items():
        single_run = run_qa_experiment([method], instances)
        relabeled[label] = single_run[method.name]
        efficiency[label] = single_run["_efficiency"]
    relabeled["_efficiency"] = efficiency
    return relabeled
