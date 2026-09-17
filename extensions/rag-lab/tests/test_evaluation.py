import json
import tempfile
import unittest
from pathlib import Path

from raglab.evaluation import (
    load_answers,
    load_dataset,
    load_questions,
    prepare_split,
    score_predictions,
    verify_manifest,
)


class EvaluationTests(unittest.TestCase):
    def dataset(self, directory: Path) -> Path:
        path = directory / "dataset.jsonl"
        rows = [
            {"id": "a", "text": "Alpha", "options": {"yes": "Yes", "no": "No"}, "answer": "yes", "group": "one"},
            {"id": "b", "text": " alpha  ", "options": {"yes": "Yes", "no": "No"}, "answer": "no", "group": "two"},
            {"id": "c", "text": "Gamma", "options": {"yes": "Yes", "no": "No"}, "answer": "yes", "group": "two"},
            {"id": "d", "text": "Delta", "options": {"yes": "Yes", "no": "No"}, "answer": "yes", "group": "four"},
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_split_is_deterministic_group_aware_and_excludes_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.dataset(root)
            first = prepare_split(source, root / "one")
            second = prepare_split(source, root / "two")
            self.assertEqual(first["counts"], second["counts"])
            for name in ("dev.questions.jsonl", "test.questions.jsonl"):
                left = (root / "one" / name).read_text()
                right = (root / "two" / name).read_text()
                self.assertEqual(left, right)
                self.assertNotIn('"answer"', left)
            test = {q.id for q in load_questions(root / "one" / "test.questions.jsonl")}
            self.assertFalse({"b", "c"} & test or ("a" in test and "b" in test))
            self.assertEqual(verify_manifest(root / "one" / "manifest.json")["version"], 1)

    def test_score_uses_all_questions_and_counts_missing(self):
        from raglab.models import Question

        questions = [Question("a", "A", {"yes": "Yes"}), Question("b", "B", {"yes": "Yes"})]
        result = score_predictions(questions, {"a": "yes", "b": "yes"}, {"a": "yes"})
        self.assertEqual((result["n"], result["correct"], result["failures"]), (2, 1, 1))
        self.assertEqual(result["accuracy"], 0.5)

    def test_corrupted_manifest_and_duplicate_prediction_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            prepare_split(self.dataset(root), output)
            (output / "dev.questions.jsonl").write_text("tampered\n")
            with self.assertRaises(ValueError):
                verify_manifest(output / "manifest.json")
            duplicate = root / "predictions.json"
            duplicate.write_text('{"x": "a", "x": "b"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                score_predictions([], {}, duplicate)

    def test_dataset_rejects_blank_and_bad_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text(json.dumps({"id": "q", "text": "Q", "options": {"a": "A"}, "answer": "b"}) + "\n")
            with self.assertRaises(ValueError):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
