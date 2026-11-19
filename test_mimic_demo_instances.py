import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nsrag.data.mimic_demo_instances import build_qa_instances_from_mimic
from nsrag.data.mimic_loader import MimicLoader

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data" / "mimic_demo_sample"


class TestMimicDemoInstances(unittest.TestCase):
    def setUp(self):
        self.loader = MimicLoader(SAMPLE_DIR)

    def test_builds_instance_with_synthesized_note_when_notes_empty(self):
        instances = build_qa_instances_from_mimic(self.loader, ["10001"])
        self.assertEqual(len(instances), 1)
        inst = instances[0]
        self.assertEqual(inst.notes[0].note_id, "synthesized_from_structured_data")
        self.assertIn("chf nos", inst.notes[0].text.lower())

    def test_reference_assertions_derived_from_real_structured_data(self):
        instances = build_qa_instances_from_mimic(self.loader, ["10001"])
        inst = instances[0]
        self.assertIn("patient has chf nos", inst.reference_assertions)
        self.assertIn("patient takes furosemide", inst.reference_assertions)

    def test_question_grounded_in_real_diagnosis(self):
        instances = build_qa_instances_from_mimic(self.loader, ["10001"])
        self.assertIn("chf nos", instances[0].question.lower())

    def test_skips_patients_with_no_structured_events(self):
        instances = build_qa_instances_from_mimic(self.loader, ["99999"])
        self.assertEqual(len(instances), 0)


if __name__ == "__main__":
    unittest.main()
