"""B3 -- Graph Linearization: patient graph information serialized into
textual context without the complete NS-RAG architecture (Section 5.4).

Reuses GraphSynth + DGsumm + HERSE + GraphRetriever (identical to
NS-RAG's retrieval) but flattens the retrieved triples into a single
text block and skips post-generation verification entirely -- this is
exactly what isolates "does structure help retrieval/grounding" (B3)
from "does the verification layer help" (B7/B8), per RQ1/RQ2/RQ5.
"""
from __future__ import annotations

from nsrag.baselines.common import Method
from nsrag.graph.dgsumm import DGsumm
from nsrag.graph.graphsynth import GraphSynth
from nsrag.graph.rdf_transform import HERSE
from nsrag.retrieval.graph_retriever import GraphRetriever
from nsrag.schema import GenerationOutput, timer


class GraphLinearization(Method):
    name = "B3 - Graph Linearization"

    def __init__(self, generator, graphsynth=None, dgsumm=None, herse=None, retriever=None):
        super().__init__(generator)
        self.graphsynth = graphsynth or GraphSynth()
        self.dgsumm = dgsumm or DGsumm()
        self.herse = herse or HERSE()
        self.retriever = retriever or GraphRetriever()

    def run(self, patient_id, notes, structured_events, question, task) -> GenerationOutput:
        with timer() as t:
            full_graph = self.graphsynth.build(patient_id, notes, structured_events)
            task_query = question or "clinical summary"
            summary_graph = self.dgsumm.summarize(full_graph, task_query)
            triples = self.herse.to_triples(summary_graph)
            selected = self.retriever.retrieve(triples, task_query)
        retrieval_time = t()

        evidence_text = self.retriever.to_linearized_context(selected)
        result, _ = self._generate(evidence_text=evidence_text, question=question)

        return GenerationOutput(
            method=self.name, patient_id=patient_id, task=task, question=question,
            answer=result.text, retrieved_context=[t.as_text() for t in selected],
            context_tokens=sum(len(t.as_text().split()) for t in selected),
            retrieval_time_s=retrieval_time, generation_time_s=result.generation_time_s,
        )
