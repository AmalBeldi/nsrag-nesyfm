"""B6 -- External Biomedical KG + LLM (Section 5.4): generation grounded
using an external biomedical knowledge graph rather than the
patient-specific graph.

This baseline needs an external KG endpoint (e.g. UMLS/SNOMED-CT via a
local graph database, or a public biomedical KG API) that is out of
scope to vendor here. `ExternalKGClient` defines the minimal interface
this baseline needs; implement it against whatever KG you have
credentialed access to and pass it in.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from nsrag.baselines.common import Method
from nsrag.schema import GenerationOutput, Triple, timer


class ExternalKGClient(ABC):
    """Minimal interface: given entity mentions, return related facts
    from an external (non-patient-specific) knowledge graph."""

    @abstractmethod
    def lookup(self, entity_mentions: list[str], k: int = 25) -> list[Triple]:
        ...


class ExternalKGPlusLLM(Method):
    name = "B6 - External KG + LLM"

    def __init__(self, generator, kg_client: ExternalKGClient, entity_extractor=None):
        super().__init__(generator)
        self.kg_client = kg_client
        if entity_extractor is None:
            from nsrag.graph.extraction import RuleBasedExtractor
            entity_extractor = RuleBasedExtractor()
        self.entity_extractor = entity_extractor

    def run(self, patient_id, notes, structured_events, question, task) -> GenerationOutput:
        with timer() as t:
            mentions: list[str] = []
            for note in notes:
                mentions.extend(e.text for e in self.entity_extractor.extract_entities(note.text))
            triples = self.kg_client.lookup(mentions)
        retrieval_time = t()

        evidence_text = ". ".join(tr.as_text() for tr in triples) or "(no external KG facts found)"
        result, _ = self._generate(evidence_text=evidence_text, question=question)

        return GenerationOutput(
            method=self.name, patient_id=patient_id, task=task, question=question,
            answer=result.text, retrieved_context=[tr.as_text() for tr in triples],
            context_tokens=sum(len(tr.as_text().split()) for tr in triples),
            retrieval_time_s=retrieval_time, generation_time_s=result.generation_time_s,
        )
