"""Clinical entity and relation extraction.

The paper (Section 5.6) specifies SciSpacy plus rule-based biomedical
relation patterns. SciSpacy models are large downloads (en_core_sci_lg,
en_ner_bc5cdr_md, ...) and are therefore not vendored here; instead this
module defines a small `EntityExtractor` / `RelationExtractor`
interface with two implementations:

  * `SciSpacyExtractor`   -- the real extractor used to produce the
                              paper's reported numbers. Requires
                              `pip install scispacy` and the relevant
                              model archives (see README.md).
  * `RuleBasedExtractor`  -- a dependency-free fallback (regex +
                              keyword lexicons) used for local testing,
                              CI, and as a documented ablation
                              ("rule-based extraction only") if you
                              want to report extraction-method
                              sensitivity as an additional robustness
                              check.

Both implementations satisfy the same interface so the rest of the
pipeline (graph construction, RQ6 extraction-robustness experiments) is
agnostic to which one is plugged in.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class EntityMention:
    text: str
    label: str          # e.g. "CONDITION", "DRUG", "PROCEDURE", "LAB"
    start: int
    end: int


@dataclass(frozen=True)
class RelationMention:
    subj: EntityMention
    pred: str            # e.g. "treats", "indicates", "administered_for"
    obj: EntityMention


class EntityExtractor(ABC):
    @abstractmethod
    def extract_entities(self, text: str) -> list[EntityMention]:
        ...


class RelationExtractor(ABC):
    @abstractmethod
    def extract_relations(self, text: str, entities: list[EntityMention]) -> list[RelationMention]:
        ...


# --------------------------------------------------------------------------- #
# SciSpacy-backed extractor (used to reproduce the paper's numbers)
# --------------------------------------------------------------------------- #

class SciSpacyExtractor(EntityExtractor):
    """Wraps a SciSpacy pipeline for clinical NER.

    Not imported eagerly: `spacy`/`scispacy` are heavy optional
    dependencies, only required when actually running experiments.
    """

    def __init__(self, model_name: str = "en_core_sci_lg"):
        try:
            import spacy  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ImportError(
                "SciSpacyExtractor requires `spacy` and `scispacy`. Install with:\n"
                "  pip install scispacy\n"
                "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/"
                "v0.5.4/en_core_sci_lg-0.5.4.tar.gz\n"
                "or use RuleBasedExtractor for a dependency-free run."
            ) from exc
        self.nlp = spacy.load(model_name)

    def extract_entities(self, text: str) -> list[EntityMention]:
        doc = self.nlp(text)
        return [
            EntityMention(text=ent.text, label=ent.label_ or "ENTITY",
                          start=ent.start_char, end=ent.end_char)
            for ent in doc.ents
        ]


# --------------------------------------------------------------------------- #
# Dependency-free fallback / baseline extractor
# --------------------------------------------------------------------------- #

# Small illustrative lexicons. Replace/extend with UMLS-derived lists
# (e.g. via QuickUMLS) for anything beyond smoke-testing the pipeline.
_DRUG_LEXICON = {
    "metformin", "insulin", "lisinopril", "atorvastatin", "amlodipine",
    "furosemide", "heparin", "warfarin", "aspirin", "levothyroxine",
}
_CONDITION_LEXICON = {
    "type 2 diabetes", "diabetes mellitus", "hypertension", "chronic kidney disease",
    "heart failure", "atrial fibrillation", "copd", "pneumonia", "sepsis", "anemia",
}
_PROCEDURE_LEXICON = {
    "hemodialysis", "intubation", "catheterization", "echocardiogram", "biopsy",
}
_LAB_PATTERN = re.compile(
    r"\b(glucose|creatinine|hemoglobin a1c|potassium|sodium|hemoglobin|wbc|platelet)"
    r"\s*(?:of|:|was|is)?\s*([\d]+\.?[\d]*)\b",
    re.IGNORECASE,
)

_REL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(started|initiated|prescribed)\b", re.IGNORECASE), "administered_for"),
    (re.compile(r"\b(diagnosed with|presents with|history of)\b", re.IGNORECASE), "has_condition"),
    (re.compile(r"\b(due to|secondary to|caused by)\b", re.IGNORECASE), "caused_by"),
    (re.compile(r"\b(treated with|managed with)\b", re.IGNORECASE), "treats"),
]


class RuleBasedExtractor(EntityExtractor, RelationExtractor):
    """Lexicon + regex extractor. No external downloads required.

    This is intentionally simple: it exists so the rest of the pipeline
    (graph construction, retrieval, verification, metrics) can be
    exercised end-to-end and unit-tested without SciSpacy models, and
    so RQ6 (extraction robustness) has a clearly weaker reference point
    to compare SciSpacy-based extraction against.
    """

    def extract_entities(self, text: str) -> list[EntityMention]:
        mentions: list[EntityMention] = []
        for lexicon, label in (
            (_DRUG_LEXICON, "DRUG"),
            (_CONDITION_LEXICON, "CONDITION"),
            (_PROCEDURE_LEXICON, "PROCEDURE"),
        ):
            for term in lexicon:
                for m in re.finditer(re.escape(term), text, re.IGNORECASE):
                    mentions.append(EntityMention(
                        text=text[m.start():m.end()], label=label,
                        start=m.start(), end=m.end(),
                    ))
        for m in _LAB_PATTERN.finditer(text):
            mentions.append(EntityMention(
                text=m.group(0), label="LAB", start=m.start(), end=m.end(),
            ))
        mentions.sort(key=lambda e: e.start)
        return mentions

    def extract_relations(self, text: str, entities: list[EntityMention]) -> list[RelationMention]:
        relations: list[RelationMention] = []
        for pattern, pred in _REL_PATTERNS:
            for m in pattern.finditer(text):
                left = [e for e in entities if e.end <= m.start()]
                right = [e for e in entities if e.start >= m.end()]
                if left and right:
                    relations.append(RelationMention(subj=left[-1], pred=pred, obj=right[0]))
        return relations
