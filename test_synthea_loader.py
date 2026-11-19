import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nsrag.data.synthea_loader import SyntheaLoader

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data" / "synthea_csv"


class TestSyntheaLoader(unittest.TestCase):
    def setUp(self):
        self.loader = SyntheaLoader(SAMPLE_DIR)

    def test_loads_expected_patient_count(self):
        patients = self.loader.load_patients()
        self.assertEqual(len(patients), 2)

    def test_conditions_and_medications_attached_per_patient(self):
        patients = {p.patient_id: p for p in self.loader.load_patients()}
        self.assertEqual(len(patients["p-0001"].conditions), 2)
        self.assertEqual(len(patients["p-0001"].medications), 1)
        self.assertEqual(len(patients["p-0002"].conditions), 1)

    def test_reference_assertions_derived_from_ground_truth(self):
        instances = self.loader.to_qa_instances()
        inst = next(i for i in instances if i.patient_id == "p-0001")
        self.assertIn("patient has diabetes mellitus type 2", inst.reference_assertions)
        self.assertIn("patient takes metformin 500 mg", inst.reference_assertions)

    def test_qa_instances_have_nonempty_notes_and_questions(self):
        instances = self.loader.to_qa_instances()
        for inst in instances:
            self.assertTrue(inst.notes[0].text)
            self.assertTrue(inst.question)

    def test_missing_csv_raises_clear_error(self):
        bad_loader = SyntheaLoader("/nonexistent/path")
        with self.assertRaises(FileNotFoundError):
            bad_loader.load_patients()


if __name__ == "__main__":
    unittest.main()
