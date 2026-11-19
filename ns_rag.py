"""B7 (NS-RAG w/o Verification) and B8 (full NS-RAG), Section 5.4.

B8 is the complete framework: graph construction -> summarization ->
RDF transformation -> graph-grounded retrieval -> generation ->
assertion extraction -> three-state verification -> minimal
explanation extraction. B7 runs the identical pipeline up to and
including generation, then skips the last two stages -- this isolates
the verification layer's contribution (RQ2/RQ3/RQ5, Table 8 row
"-- symbolic verification").
"""
from __future__ import annotations

from nsrag.baselines.common import Method
from nsrag.graph.dgsumm import DGsumm
from nsrag.graph.graphsynth import GraphSynth
from nsrag.graph.rdf_transform import HERSE
from nsrag.retrieval.graph_retriever import GraphRetriever
from nsrag.schema import GenerationOutput, timer
from nsrag.verification.assertion_extraction import extract_assertions
from nsrag.verification.explain import extract_minimal_explanation, minimality_ratio
from nsrag.verification.verifier import GraphVerifier


class _NSRAGBase(Method):
    def __init__(self, generator, graphsynth=None, dgsumm=None, herse=None, retriever=None):
        super().__init__(generator)
        self.graphsynth = graphsynth or GraphSynth()
        self.dgsumm = dgsumm or DGsumm()
        self.herse = herse or HERSE()
        self.retriever = retriever or GraphRetriever()

    def _retrieve(self, patient_id, notes, structured_events, question):
        with timer() as t:
            full_graph = self.graphsynth.build(patient_id, notes, structured_events)
            task_query = question or "clinical summary"
            summary_graph = self.dgsumm.summarize(full_graph, task_query)
            triples = self.herse.to_triples(summary_graph)
            selected = self.retriever.retrieve(triples, task_query)
        return selected, t()


class NSRAGWithoutVerification(_NSRAGBase):
    name = "B7 - NS-RAG w/o Verification"

    def run(self, patient_id, notes, structured_events, question, task) -> GenerationOutput:
        selected, retrieval_time = self._retrieve(patient_id, notes, structured_events, question)
        evidence_text = "\n".join(GraphRetriever.to_structured_context(selected))
        result, _ = self._generate(evidence_text=evidence_text, question=question)

        return GenerationOutput(
            method=self.name, patient_id=patient_id, task=task, question=question,
            answer=result.text, retrieved_context=[t.as_text() for t in selected],
            context_tokens=sum(len(t.as_text().split()) for t in selected),
            retrieval_time_s=retrieval_time, generation_time_s=result.generation_time_s,
        )


class NSRAG(_NSRAGBase):
    name = "B8 - NS-RAG"

    def __init__(self, generator, verifier=None, **kwargs):
        super().__init__(generator, **kwargs)
        self.verifier = verifier or GraphVerifier()

    def run(self, patient_id, notes, structured_events, question, task) -> GenerationOutput:
        selected, retrieval_time = self._retrieve(patient_id, notes, structured_events, question)
        evidence_text = "\n".join(GraphRetriever.to_structured_context(selected))
        result, _ = self._generate(evidence_text=evidence_text, question=question)

        assertions = extract_assertions(result.text, id_prefix=f"{patient_id}_a")
        assertions = self.verifier.verify_all(assertions, selected)

        min_graphs = {}
        for a in assertions:
            if a.witness_subgraph:
                g_min = extract_minimal_explanation(a, a.witness_subgraph, self.verifier)
                min_graphs[a.id] = g_min
                a.minimality_ratio = minimality_ratio(a.witness_subgraph, g_min)

        return GenerationOutput(
            method=self.name, patient_id=patient_id, task=task, question=question,
            answer=result.text, retrieved_context=[t.as_text() for t in selected],
            context_tokens=sum(len(t.as_text().split()) for t in selected),
            retrieval_time_s=retrieval_time, generation_time_s=result.generation_time_s,
            assertions=assertions,
            minimal_explanation_graphs=min_graphs,
        )
