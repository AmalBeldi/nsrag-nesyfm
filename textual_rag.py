"""B2 -- Textual RAG: retrieval of patient-specific textual evidence
followed by generation (Section 5.4)."""
from __future__ import annotations

from nsrag.baselines.common import Method
from nsrag.retrieval.text_retriever import TextRetriever, TextRetrievalConfig
from nsrag.schema import GenerationOutput, timer


class TextualRAG(Method):
    name = "B2 - Textual RAG"

    def __init__(self, generator, config: TextRetrievalConfig | None = None):
        super().__init__(generator)
        self.retriever = TextRetriever(config)

    def run(self, patient_id, notes, structured_events, question, task) -> GenerationOutput:
        note_texts = [n.text for n in notes]
        with timer() as t:
            chunks = self.retriever.retrieve(note_texts, question or "clinical summary")
        retrieval_time = t()

        evidence_text = " ".join(chunks) if chunks else "(no relevant text retrieved)"
        result, _ = self._generate(evidence_text=evidence_text, question=question)

        return GenerationOutput(
            method=self.name, patient_id=patient_id, task=task, question=question,
            answer=result.text, retrieved_context=chunks,
            context_tokens=sum(len(c.split()) for c in chunks),
            retrieval_time_s=retrieval_time, generation_time_s=result.generation_time_s,
        )
