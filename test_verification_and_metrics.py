import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nsrag.metrics.factuality import (
    factuality_score,
    verification_macro_f1,
    verification_partition,
)
from nsrag.retrieval.graph_retriever import GraphRetriever, RetrievalConfig
from nsrag.retrieval.text_retriever import TextRetriever
from nsrag.schema import Assertion, Triple, VerificationState
from nsrag.verification.assertion_extraction import extract_assertions
from nsrag.verification.explain import extract_minimal_explanation, minimality_ratio
from nsrag.verification.verifier import GraphVerifier


class TestVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = GraphVerifier()
        self.evidence = [
            Triple("Patient", "has_condition", "type 2 diabetes"),
            Triple("Patient", "takes", "metformin"),
            Triple("Patient", "has_condition", "chest pain"),
        ]

    def test_supported_assertion(self):
        a = Assertion(id="a1", text="patient has type 2 diabetes")
        result = self.verifier.verify(a, self.evidence)
        self.assertEqual(result.state, VerificationState.SUPPORTED)
        self.assertTrue(result.witness_subgraph)

    def test_contradicted_assertion_via_negation_mismatch(self):
        a = Assertion(id="a2", text="the patient denies chest pain")
        result = self.verifier.verify(a, self.evidence)
        self.assertEqual(result.state, VerificationState.CONTRADICTED)

    def test_unknown_assertion_with_no_matching_evidence(self):
        a = Assertion(id="a3", text="patient underwent hemodialysis")
        result = self.verifier.verify(a, self.evidence)
        self.assertEqual(result.state, VerificationState.UNKNOWN)

    def test_verify_all_preserves_order(self):
        assertions = [
            Assertion(id="a1", text="patient has type 2 diabetes"),
            Assertion(id="a2", text="patient underwent hemodialysis"),
        ]
        results = self.verifier.verify_all(assertions, self.evidence)
        self.assertEqual([r.id for r in results], ["a1", "a2"])


class TestExplanation(unittest.TestCase):
    def test_minimal_explanation_is_subset_and_still_verifies(self):
        verifier = GraphVerifier()
        evidence = [
            Triple("Patient", "has_condition", "type 2 diabetes"),
            Triple("Patient", "has_condition", "hypertension"),   # irrelevant filler
            Triple("Patient", "has_condition", "anemia"),          # irrelevant filler
        ]
        a = Assertion(id="a1", text="patient has type 2 diabetes")
        a = verifier.verify(a, evidence)
        self.assertEqual(a.state, VerificationState.SUPPORTED)

        g_min = extract_minimal_explanation(a, a.witness_subgraph, verifier)
        self.assertLessEqual(len(g_min), len(a.witness_subgraph))
        self.assertGreaterEqual(len(g_min), 1)

        mr = minimality_ratio(a.witness_subgraph, g_min)
        self.assertGreaterEqual(mr, 0.0)
        self.assertLessEqual(mr, 1.0)

    def test_minimality_ratio_empty_graph_is_zero(self):
        self.assertEqual(minimality_ratio([], []), 0.0)


class TestAssertionExtraction(unittest.TestCase):
    def test_splits_conjoined_sentence_into_clauses(self):
        text = "The patient has diabetes and was started on metformin."
        assertions = extract_assertions(text)
        self.assertGreaterEqual(len(assertions), 2)

    def test_empty_text_yields_no_assertions(self):
        self.assertEqual(extract_assertions(""), [])


class TestMetrics(unittest.TestCase):
    def test_factuality_perfect_match(self):
        gen = [Assertion(id="a1", text="patient has diabetes")]
        ref = ["patient has diabetes"]
        score = factuality_score(gen, ref)
        self.assertEqual(score.precision, 1.0)
        self.assertEqual(score.recall, 1.0)
        self.assertEqual(score.f1, 1.0)

    def test_factuality_no_overlap(self):
        gen = [Assertion(id="a1", text="patient has diabetes")]
        ref = ["patient underwent hemodialysis"]
        score = factuality_score(gen, ref)
        self.assertEqual(score.f1, 0.0)

    def test_verification_partition_sums_to_one(self):
        assertions = [
            Assertion(id=str(i), text="x", state=s)
            for i, s in enumerate([
                VerificationState.SUPPORTED, VerificationState.SUPPORTED,
                VerificationState.CONTRADICTED, VerificationState.UNKNOWN,
            ])
        ]
        part = verification_partition(assertions)
        self.assertAlmostEqual(part.ec + part.cr + part.ur, 1.0, places=6)
        self.assertEqual(part.ec, 0.5)

    def test_verification_partition_empty(self):
        part = verification_partition([])
        self.assertEqual((part.ec, part.cr, part.ur), (0.0, 0.0, 0.0))

    def test_macro_f1_perfect_predictions(self):
        labels = [VerificationState.SUPPORTED, VerificationState.CONTRADICTED, VerificationState.UNKNOWN]
        result = verification_macro_f1(labels, labels)
        self.assertEqual(result["macro_f1"], 1.0)


class TestRetrieval(unittest.TestCase):
    def test_graph_retriever_respects_token_budget(self):
        triples = [Triple("Patient", "has_condition", f"condition_{i}") for i in range(50)]
        retriever = GraphRetriever(RetrievalConfig(k=50, max_context_tokens=10))
        selected = retriever.retrieve(triples, "condition_1")
        total_tokens = sum(len(t.as_text().split()) for t in selected)
        self.assertLessEqual(total_tokens, 10)

    def test_graph_retriever_ranks_relevant_triples_first(self):
        triples = [
            Triple("Patient", "has_condition", "hypertension"),
            Triple("Patient", "has_condition", "type 2 diabetes"),
        ]
        retriever = GraphRetriever(RetrievalConfig(k=1, max_context_tokens=1000))
        selected = retriever.retrieve(triples, "does patient have diabetes")
        self.assertEqual(selected[0].obj, "type 2 diabetes")

    def test_text_retriever_returns_relevant_chunks(self):
        notes = ["Patient has type 2 diabetes. Glucose was 210 today. Unrelated sentence about weather."]
        retriever = TextRetriever()
        selected = retriever.retrieve(notes, "glucose level")
        self.assertTrue(any("glucose" in c.lower() or "210" in c for c in selected))


if __name__ == "__main__":
    unittest.main()



class TestOpenWorldVerification(unittest.TestCase):
    def setUp(self):
        from nsrag.schema import Triple
        from nsrag.verification.verifier import GraphVerifier
        self.v = GraphVerifier()
        self.ev = [Triple("Patient", "has_condition", "type 2 diabetes mellitus"),
                   Triple("Patient", "takes_medication", "metformin 500 mg"),
                   Triple("Patient", "has_condition", "hypertension")]

    def _state(self, text, ev=None):
        from nsrag.schema import Assertion
        return self.v.verify(Assertion(id="x", text=text), ev or self.ev).state.value

    def test_absent_medication_is_unknown(self):
        self.assertEqual(self._state("patient takes insulin glargine"), "UNKNOWN")

    def test_absent_condition_is_unknown(self):
        self.assertEqual(self._state("patient has asthma"), "UNKNOWN")

    def test_negation_still_contradicts(self):
        self.assertEqual(self._state("patient has no hypertension"), "CONTRADICTED")

    def test_supported(self):
        self.assertEqual(self._state("patient takes metformin 500 mg"), "SUPPORTED")

    def test_functional_lab_conflict_still_contradicted(self):
        from nsrag.schema import Triple
        ev = [Triple("Patient", "lab_HbA1c", "8.2 percent")]
        self.assertEqual(self._state("patient lab HbA1c 6.0 percent", ev), "CONTRADICTED")

    def test_lab_value_mismatch_not_supported(self):
        from nsrag.schema import Triple
        ev = [Triple("Patient", "lab_creatinine", "1.1 mg/dl")]
        self.assertEqual(self._state("patient lab creatinine 2.4 mg/dl", ev), "CONTRADICTED")

    def test_absent_analyte_is_unknown(self):
        from nsrag.schema import Triple
        ev = [Triple("Patient", "lab_HbA1c", "8.2 percent")]
        self.assertEqual(self._state("patient lab sodium 140 mmol/l", ev), "UNKNOWN")
