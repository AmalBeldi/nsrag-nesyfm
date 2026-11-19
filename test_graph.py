import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nsrag.graph.dgsumm import DGsumm, DGsummConfig
from nsrag.graph.extraction import RuleBasedExtractor
from nsrag.graph.graphsynth import ClinicalNote, GraphSynth, StructuredEvent
from nsrag.graph.rdf_transform import HERSE


class TestExtraction(unittest.TestCase):
    def setUp(self):
        self.extractor = RuleBasedExtractor()

    def test_entity_extraction_finds_known_terms(self):
        text = "Patient diagnosed with type 2 diabetes. Started metformin."
        ents = self.extractor.extract_entities(text)
        labels = {(e.text.lower(), e.label) for e in ents}
        self.assertIn(("type 2 diabetes", "CONDITION"), labels)
        self.assertIn(("metformin", "DRUG"), labels)

    def test_relation_extraction_links_entities(self):
        text = "Diagnosed with hypertension. Treated with amlodipine."
        ents = self.extractor.extract_entities(text)
        rels = self.extractor.extract_relations(text, ents)
        self.assertTrue(any(r.pred == "treats" for r in rels))


class TestGraphSynth(unittest.TestCase):
    def test_build_produces_nodes_and_temporal_edges(self):
        gs = GraphSynth()
        notes = [ClinicalNote(patient_id="p1", note_id="n1",
                               text="Diagnosed with type 2 diabetes.", timestamp="2024-01-01")]
        events = [
            StructuredEvent(patient_id="p1", event_type="lab", label="Glucose",
                             value="210", timestamp="2024-01-02"),
            StructuredEvent(patient_id="p1", event_type="medication", label="Metformin",
                             value="500mg", timestamp="2024-01-03"),
        ]
        graph = gs.build("p1", notes=notes, structured_events=events)
        self.assertGreaterEqual(graph.n_nodes(), 3)
        # structured events should be temporally chained
        precedes_edges = [e for e in graph.edges if e.pred == "precedes"]
        self.assertEqual(len(precedes_edges), 1)

    def test_build_is_deterministic(self):
        gs = GraphSynth()
        notes = [ClinicalNote(patient_id="p1", note_id="n1",
                               text="Diagnosed with hypertension.", timestamp="2024-01-01")]
        g1 = gs.build("p1", notes=notes, structured_events=[])
        g2 = gs.build("p1", notes=notes, structured_events=[])
        self.assertEqual(set(g1.nodes.keys()), set(g2.nodes.keys()))


class TestDGsumm(unittest.TestCase):
    def test_summarize_reduces_graph_size(self):
        gs = GraphSynth()
        notes = [ClinicalNote(
            patient_id="p1", note_id="n1",
            text=("Diagnosed with type 2 diabetes and hypertension and chronic kidney disease. "
                  "Started metformin and lisinopril and furosemide. "
                  "Treated with insulin due to poor control."),
            timestamp="2024-01-01",
        )]
        graph = gs.build("p1", notes=notes, structured_events=[])
        dgsumm = DGsumm(DGsummConfig(keep_ratio=0.3, min_nodes=1))
        summary = dgsumm.summarize(graph, task_query="glucose control plan")

        self.assertLessEqual(summary.n_nodes(), graph.n_nodes())
        stats = DGsumm.reduction_stats(graph, summary)
        self.assertGreaterEqual(stats["nodes_reduction_pct"], 0.0)

    def test_summarize_never_produces_dangling_edges(self):
        gs = GraphSynth()
        notes = [ClinicalNote(patient_id="p1", note_id="n1",
                               text="Diagnosed with hypertension. Treated with amlodipine.",
                               timestamp="2024-01-01")]
        graph = gs.build("p1", notes=notes, structured_events=[])
        summary = DGsumm(DGsummConfig(keep_ratio=0.5, min_nodes=1)).summarize(graph, "amlodipine")
        node_ids = set(summary.nodes.keys())
        for edge in summary.edges:
            self.assertIn(edge.subj, node_ids)
            self.assertIn(edge.obj, node_ids)


class TestHERSE(unittest.TestCase):
    def test_to_triples_covers_all_nodes_and_edges(self):
        gs = GraphSynth()
        notes = [ClinicalNote(patient_id="p1", note_id="n1",
                               text="Diagnosed with hypertension.", timestamp="2024-01-01")]
        graph = gs.build("p1", notes=notes, structured_events=[])
        triples = HERSE().to_triples(graph)
        # at least a rdf:type + hasText triple per node
        self.assertGreaterEqual(len(triples), 2 * graph.n_nodes())


if __name__ == "__main__":
    unittest.main()
