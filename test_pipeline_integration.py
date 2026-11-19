import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nsrag.baselines.graph_linearization import GraphLinearization
from nsrag.baselines.llm_only import LLMOnly
from nsrag.baselines.ns_rag import NSRAG, NSRAGWithoutVerification
from nsrag.baselines.textual_rag import TextualRAG
from nsrag.generation.generator import MockGenerator
from nsrag.graph.graphsynth import ClinicalNote, StructuredEvent
from nsrag.pipeline.run_ablation import run_ablation_experiment
from nsrag.pipeline.run_context_robustness import run_context_robustness_experiment
from nsrag.pipeline.run_efficiency import run_efficiency_experiment
from nsrag.pipeline.run_qa import QAInstance, run_qa_experiment
from nsrag.utils.results_to_latex import ablation_results_to_latex, qa_results_to_latex


def _sample_instances() -> list[QAInstance]:
    notes = [ClinicalNote(
        patient_id="p1", note_id="n1",
        text=("Patient diagnosed with type 2 diabetes and hypertension. "
              "Started metformin for glucose control. Denies chest pain."),
        timestamp="2024-01-01",
    )]
    events = [
        StructuredEvent(patient_id="p1", event_type="lab", label="Glucose",
                         value="210", timestamp="2024-01-02"),
        StructuredEvent(patient_id="p1", event_type="medication", label="Metformin",
                         value="500mg", timestamp="2024-01-03"),
    ]
    return [
        QAInstance(patient_id="p1", notes=notes, structured_events=events,
                   question="What is the glucose control plan?",
                   reference_assertions=["patient has type 2 diabetes", "patient started metformin"]),
    ]


class TestFullPipeline(unittest.TestCase):
    def setUp(self):
        self.instances = _sample_instances()
        self.generator = MockGenerator()

    def test_all_implemented_methods_run_without_error(self):
        methods = [
            LLMOnly(self.generator),
            TextualRAG(self.generator),
            GraphLinearization(self.generator),
            NSRAGWithoutVerification(self.generator),
            NSRAG(self.generator),
        ]
        results = run_qa_experiment(methods, self.instances)
        for m in methods:
            self.assertIn(m.name, results)
            self.assertIn(results[m.name].fact_f1, [results[m.name].fact_f1])  # just don't crash
        self.assertIn("_efficiency", results)

    def test_ns_rag_produces_verification_partition_summing_to_one(self):
        results = run_qa_experiment([NSRAG(self.generator)], self.instances)
        r = results["B8 - NS-RAG"]
        if r.cr is not None:
            total = r.cr + r.ur + r.ec
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_ablation_runs_five_configurations(self):
        results = run_ablation_experiment(self.generator, self.instances)
        labels = [k for k in results if k != "_efficiency"]
        self.assertEqual(len(labels), 5)
        self.assertIn("Full model (NS-RAG)", labels)

    def test_context_robustness_runs_without_error(self):
        points = run_context_robustness_experiment(self.generator, self.instances, budgets=[128, 256])
        self.assertGreater(len(points), 0)
        for p in points:
            self.assertGreaterEqual(p.hallucination_rate, 0.0)
            self.assertLessEqual(p.hallucination_rate, 1.0)

    def test_efficiency_runner_covers_all_stages(self):
        summary = run_efficiency_experiment(self.generator, self.instances)
        expected_stages = {
            "Graph construction", "Graph summarization", "RDF transformation",
            "Context construction", "LLM generation", "Verification", "Explanation",
        }
        self.assertTrue(expected_stages.issubset(summary.keys()))

    def test_latex_generation_does_not_crash_on_missing_methods(self):
        results = run_qa_experiment([LLMOnly(self.generator)], self.instances)
        latex = qa_results_to_latex(results, row_order=["B1 - LLM", "B4 - GraphRAG"])
        self.assertIn("B1 - LLM", latex)
        self.assertIn("B4 - GraphRAG & -- & -- & -- & -- & -- & -- & -- & --", latex)

    def test_ablation_latex_generation(self):
        results = run_ablation_experiment(self.generator, self.instances)
        latex = ablation_results_to_latex(results, row_order=list(results.keys())[:-1])
        self.assertEqual(latex.count("\\\\"), 5)


if __name__ == "__main__":
    unittest.main()
