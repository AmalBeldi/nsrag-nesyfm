"""Clinical summarization runner (Table 6, Section 5.12).

Structurally identical to `run_qa.py` but with `question=None` (each
`Method.run` then generates a full patient summary instead of
answering a specific question, per the `Method` interface contract)
and with ROUGE-L/BERTScore added against reference discharge
summaries, since summarization quality also depends on coverage and
fluency, not just atomic factuality (Section 5.8, "Text Similarity").
"""
from __future__ import annotations

from dataclasses import dataclass, field

from nsrag.baselines.common import Method
from nsrag.graph.graphsynth import ClinicalNote, StructuredEvent
from nsrag.metrics.factuality import factuality_score, verification_partition
from nsrag.metrics.text_similarity import bertscore_corpus, rouge_l_corpus
from nsrag.schema import GenerationOutput
from nsrag.verification.assertion_extraction import extract_assertions


@dataclass
class SummarizationInstance:
    patient_id: str
    notes: list[ClinicalNote]
    structured_events: list[StructuredEvent]
    reference_summary: str
    reference_assertions: list[str] = field(default_factory=list)


@dataclass
class SummarizationResult:
    method: str
    fact_f1: float
    cr: float | None
    ur: float | None
    ec: float | None
    rouge_l: float | None
    bertscore_f1: float | None
    outputs: list[GenerationOutput] = field(default_factory=list)


def run_summarization_experiment(
    methods: list[Method],
    instances: list[SummarizationInstance],
    compute_text_similarity: bool = True,
    match_threshold: float = 0.5,
) -> dict[str, SummarizationResult]:
    results: dict[str, SummarizationResult] = {}

    for method in methods:
        outputs, fact_scores, all_assertions = [], [], []
        predictions, references = [], []

        for inst in instances:
            output = method.run(
                inst.patient_id, inst.notes, inst.structured_events,
                question=None, task="summarization",
            )
            outputs.append(output)
            predictions.append(output.answer)
            references.append(inst.reference_summary)

            gen_assertions = output.assertions or extract_assertions(output.answer)
            if inst.reference_assertions:
                fact_scores.append(
                    factuality_score(gen_assertions, inst.reference_assertions, match_threshold)
                )
            if output.assertions and any(a.state is not None for a in output.assertions):
                all_assertions.extend(output.assertions)

        n = max(len(fact_scores), 1)
        f1 = sum(s.f1 for s in fact_scores) / n if fact_scores else 0.0

        cr = ur = ec = None
        if all_assertions:
            part = verification_partition(all_assertions)
            cr, ur, ec = part.cr, part.ur, part.ec

        rouge_l_val = bert_f1 = None
        if compute_text_similarity:
            try:
                rouge_l_val = rouge_l_corpus(predictions, references)
                bert_f1 = bertscore_corpus(predictions, references)["f1"]
            except ImportError:
                pass  # rouge-score / bert-score not installed; leave as None

        results[method.name] = SummarizationResult(
            method=method.name, fact_f1=round(f1, 4), cr=cr, ur=ur, ec=ec,
            rouge_l=rouge_l_val, bertscore_f1=bert_f1, outputs=outputs,
        )

    return results
