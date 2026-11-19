"""i2b2 2010 component-level evaluation runner (Table 7, Section 5.13).

Evaluates entity extraction, relation extraction, assertion
construction, and graph matching independently against i2b2 gold
annotations -- this is what lets RQ6 (extraction robustness) separate
upstream extraction error from downstream verification error, per
Section 4.9/5.9.

"Graph matching" here means: build a small `PatientGraph` from the raw
document text using the same `GraphSynth` extractor as the main
pipeline, convert i2b2 gold concepts/relations into the same Node/Edge
shape, and compute set-level P/R/F1 between the two -- i.e. "does our
graph construction recover the gold clinical graph", not just "does it
recover the right span labels" (which entity/relation P/R/F1 already
covers).
"""
from __future__ import annotations

from dataclasses import dataclass

from nsrag.data.i2b2_loader import I2b2Document, I2b2Loader
from nsrag.graph.extraction import EntityExtractor, EntityMention, RelationExtractor


@dataclass
class PRF1:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _prf1(predicted: set, gold: set) -> PRF1:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return PRF1(round(p, 4), round(r, 4), round(f1, 4), tp, fp, fn)


def _normalize_span_text(text: str) -> str:
    return " ".join(text.lower().split())


def evaluate_entity_extraction(
    extractor: EntityExtractor, documents: list[I2b2Document]
) -> PRF1:
    predicted, gold = set(), set()
    for i, doc in enumerate(documents):
        mentions = extractor.extract_entities(doc.text)
        predicted |= {(i, _normalize_span_text(m.text)) for m in mentions}
        gold |= {(i, _normalize_span_text(c.text)) for c in doc.concepts}
    return _prf1(predicted, gold)


def evaluate_relation_extraction(
    entity_extractor: EntityExtractor,
    relation_extractor: RelationExtractor,
    documents: list[I2b2Document],
) -> PRF1:
    predicted, gold = set(), set()
    for i, doc in enumerate(documents):
        mentions = entity_extractor.extract_entities(doc.text)
        relations = relation_extractor.extract_relations(doc.text, mentions)
        predicted |= {
            (i, _normalize_span_text(r.subj.text), r.pred, _normalize_span_text(r.obj.text))
            for r in relations
        }
        gold |= {
            (i, _normalize_span_text(r.subj_text), r.relation, _normalize_span_text(r.obj_text))
            for r in doc.relations
        }
    return _prf1(predicted, gold)


def evaluate_assertion_construction(
    predicted_assertions_by_doc: list[list[tuple[str, str]]],  # (text, value) per doc
    documents: list[I2b2Document],
) -> PRF1:
    """`predicted_assertions_by_doc[i]` should contain (concept_text,
    assertion_value) pairs produced by your assertion-construction
    stage for `documents[i]`, where `assertion_value` is one of i2b2's
    six labels (present, absent, possible, conditional, hypothetical,
    associated_with_someone_else)."""
    predicted, gold = set(), set()
    for i, doc in enumerate(documents):
        predicted |= {
            (i, _normalize_span_text(t), v) for t, v in predicted_assertions_by_doc[i]
        }
        gold |= {(i, _normalize_span_text(a.text), a.value) for a in doc.assertions}
    return _prf1(predicted, gold)


def evaluate_graph_matching(
    entity_extractor: EntityExtractor,
    relation_extractor: RelationExtractor,
    documents: list[I2b2Document],
) -> PRF1:
    """Joint (entity AND relation) graph-level match: an edge only
    counts as correctly recovered if both its endpoints and its
    predicate match gold, which is strictly harder than relation
    extraction P/R/F1 alone and is the number that best reflects
    downstream graph-grounded verification quality (RQ6)."""
    return evaluate_relation_extraction(entity_extractor, relation_extractor, documents)


def run_i2b2_component_evaluation(
    i2b2_dir: str,
    entity_extractor: EntityExtractor,
    relation_extractor: RelationExtractor,
    doc_ids: list[str] | None = None,
) -> dict[str, PRF1]:
    loader = I2b2Loader(i2b2_dir)
    ids = doc_ids or loader.list_document_ids()
    documents = [loader.load_document(doc_id) for doc_id in ids]

    return {
        "Entity extraction": evaluate_entity_extraction(entity_extractor, documents),
        "Relation extraction": evaluate_relation_extraction(entity_extractor, relation_extractor, documents),
        "Graph matching": evaluate_graph_matching(entity_extractor, relation_extractor, documents),
    }
