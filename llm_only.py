"""B1 -- LLM: generation without retrieved external context (Section 5.4)."""
from __future__ import annotations

from typing import Optional

from nsrag.baselines.common import Method
from nsrag.graph.graphsynth import ClinicalNote, StructuredEvent
from nsrag.schema import GenerationOutput


class LLMOnly(Method):
    name = "B1 - LLM"

    def run(self, patient_id, notes, structured_events, question, task) -> GenerationOutput:
        result, _ = self._generate(evidence_text="(no external context provided)", question=question)
        return GenerationOutput(
            method=self.name, patient_id=patient_id, task=task, question=question,
            answer=result.text, retrieved_context=[], context_tokens=0,
            retrieval_time_s=0.0, generation_time_s=result.generation_time_s,
        )
