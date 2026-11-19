"""Text-similarity metrics for the summarization task (Table 6):
ROUGE-L and BERTScore, reported as "complementary text-similarity
metrics" alongside factuality/verification (Section 5.8).

Thin wrappers around `rouge-score` and `bert-score` (both optional
dependencies, only needed when actually computing these numbers).
"""
from __future__ import annotations


def rouge_l(prediction: str, reference: str) -> float:
    try:
        from rouge_score import rouge_scorer  # type: ignore
    except ImportError as exc:
        raise ImportError("pip install rouge-score") from exc

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].fmeasure


def rouge_l_corpus(predictions: list[str], references: list[str]) -> float:
    scores = [rouge_l(p, r) for p, r in zip(predictions, references)]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def bertscore_corpus(
    predictions: list[str],
    references: list[str],
    lang: str = "en",
    model_type: str | None = None,
) -> dict:
    try:
        from bert_score import score as bertscore  # type: ignore
    except ImportError as exc:
        raise ImportError("pip install bert-score") from exc

    P, R, F1 = bertscore(predictions, references, lang=lang, model_type=model_type)
    return {
        "precision": round(P.mean().item(), 4),
        "recall": round(R.mean().item(), 4),
        "f1": round(F1.mean().item(), 4),
    }
