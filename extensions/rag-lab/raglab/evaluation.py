"""Dataset preparation and honest, leakage-resistant evaluation utilities."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from pathlib import Path
from typing import Any, Mapping

from .models import Question

_QUESTION_FIELDS = {"id", "text", "options", "corpus_ids", "release"}
_ANSWER_FIELDS = {"question_id", "answer"}


def _json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank input at {path}:{line_no}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"record at {path}:{line_no} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    return rows


def _text_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate JSONL records with ``id``, public fields, and gold ``answer``."""
    records = _json_lines(Path(path))
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(records, 1):
        identifier = row.get("id")
        text = row.get("text")
        options = row.get("options")
        answer = row.get("answer")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"record {index} has a blank or invalid id")
        if identifier in seen:
            raise ValueError(f"duplicate question id: {identifier}")
        seen.add(identifier)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"record {identifier} has blank text")
        if not isinstance(options, dict) or not options:
            raise ValueError(f"record {identifier} has invalid options")
        if any(not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip()
               for key, value in options.items()):
            raise ValueError(f"record {identifier} has invalid options")
        if not isinstance(answer, str) or answer not in options:
            raise ValueError(f"record {identifier} has invalid gold answer")
        group = row.get("group")
        if group is not None and (not isinstance(group, str) or not group.strip()):
            raise ValueError(f"record {identifier} has invalid group")
        corpus_ids = row.get("corpus_ids", [])
        if not isinstance(corpus_ids, list) or any(not isinstance(item, str) or not item.strip() for item in corpus_ids):
            raise ValueError(f"record {identifier} has invalid corpus_ids")
        release = row.get("release")
        if release is not None and not isinstance(release, str):
            raise ValueError(f"record {identifier} has invalid release")
        result.append({
            "id": identifier, "text": text, "options": dict(options), "answer": answer,
            "group": group, "corpus_ids": list(corpus_ids), "release": release,
        })
    return result


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def prepare_split(dataset_path: str | Path, output_dir: str | Path, *, seed: int = 20260916,
                  test_fraction: float = 0.3) -> dict[str, Any]:
    """Create deterministic group-aware dev/test JSONL files and return a manifest."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    records = load_dataset(dataset_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = ["dev.questions.jsonl", "dev.answers.jsonl", "test.questions.jsonl", "test.answers.jsonl", "manifest.json"]
    existing = [output / name for name in names if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite: {existing[0]}")

    union = _UnionFind(len(records))
    keys: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        link_keys = [("text", _text_key(record["text"]))]
        if record["group"] is not None:
            link_keys.append(("group", _text_key(record["group"])))
        for key in link_keys:
            if key in keys:
                union.join(index, keys[key])
            else:
                keys[key] = index
    components: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        components.setdefault(union.find(index), []).append(record)
    ordered = sorted(
        components.values(),
        key=lambda rows: hashlib.sha256(
            (f"{seed}:" + "\x1f".join(sorted(row["id"] for row in rows))).encode()
        ).hexdigest(),
    )
    target = max(1, min(len(records) - 1, round(len(records) * test_fraction)))
    test: list[dict[str, Any]] = []
    for component in ordered:
        if test and len(test) + len(component) > len(records) - 1:
            continue
        if len(test) < target or not test:
            test.extend(component)
    if not test or len(test) == len(records):
        raise ValueError("cannot create non-empty dev and test partitions")
    test_ids = {row["id"] for row in test}
    splits = {"test": [row for row in records if row["id"] in test_ids],
              "dev": [row for row in records if row["id"] not in test_ids]}

    files: dict[str, dict[str, Any]] = {}
    for split, rows in splits.items():
        question_path, answer_path = output / f"{split}.questions.jsonl", output / f"{split}.answers.jsonl"
        with question_path.open("x", encoding="utf-8") as qh, answer_path.open("x", encoding="utf-8") as ah:
            for row in rows:
                qh.write(json.dumps({key: row[key] for key in _QUESTION_FIELDS}, sort_keys=True, ensure_ascii=False) + "\n")
                ah.write(json.dumps({"question_id": row["id"], "answer": row["answer"]}, sort_keys=True) + "\n")
        for path in (question_path, answer_path):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files[path.name] = {"path": path.name, "sha256": digest, "count": len(rows)}
    manifest = {"version": 1, "seed": seed, "test_fraction": test_fraction,
                "source": str(Path(dataset_path)), "counts": {key: len(value) for key, value in splits.items()},
                "files": files}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_questions(path: str | Path) -> list[Question]:
    """Load public question JSONL and reject gold or unknown fields."""
    rows = _json_lines(Path(path))
    result = []
    for row in rows:
        if set(row) - _QUESTION_FIELDS or not _QUESTION_FIELDS.issuperset(row):
            raise ValueError("question file contains non-public fields")
        result.append(Question(id=row["id"], text=row["text"], options=dict(row.get("options", {})),
                               corpus_ids=tuple(row.get("corpus_ids", [])), release=row.get("release")))
    if len({q.id for q in result}) != len(result):
        raise ValueError("duplicate question id")
    return result


def load_answers(path: str | Path) -> dict[str, str]:
    """Load answer JSONL into a question-id to option-id mapping."""
    rows = _json_lines(Path(path))
    answers: dict[str, str] = {}
    for row in rows:
        if set(row) != _ANSWER_FIELDS or not isinstance(row.get("question_id"), str) or not isinstance(row.get("answer"), str):
            raise ValueError("invalid answer record")
        if row["question_id"] in answers:
            raise ValueError("duplicate answer id")
        answers[row["question_id"]] = row["answer"]
    return answers


def _prediction_map(predictions: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(predictions, (str, Path)):
        value = json.loads(Path(predictions).read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    else:
        value = predictions
    if not isinstance(value, Mapping):
        raise ValueError("predictions must be a JSON object")
    return dict(value)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate prediction id: {key}")
        result[key] = value
    return result


def score_predictions(questions: list[Question], answers: Mapping[str, str],
                      predictions: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Score predictions with every question in the denominator."""
    question_map = {question.id: question for question in questions}
    if len(question_map) != len(questions):
        raise ValueError("duplicate question id")
    if set(answers) - set(question_map):
        raise ValueError("answer contains unknown question id")
    for identifier, answer in answers.items():
        if not isinstance(answer, str) or answer not in question_map[identifier].options:
            raise ValueError(f"invalid answer option for {identifier}")
    prediction_map = _prediction_map(predictions)
    if set(prediction_map) - set(question_map):
        raise ValueError("prediction contains unknown question id")
    correct = answered = failures = abstentions = 0
    for identifier, question in question_map.items():
        value = prediction_map.get(identifier, "__missing__")
        if value == "__missing__":
            failures += 1
            continue
        if value is None or (isinstance(value, Mapping) and value.get("abstained")):
            abstentions += 1
            continue
        selected = value.get("selected_option") if isinstance(value, Mapping) else value
        if isinstance(value, Mapping) and value.get("failed"):
            failures += 1
            continue
        if not isinstance(selected, str) or selected not in question.options:
            raise ValueError(f"invalid prediction option for {identifier}")
        answered += 1
        correct += int(answers.get(identifier) == selected)
    n = len(questions)
    accuracy = correct / n
    coverage = answered / n
    denominator = 1 + 1.96**2 / n
    centre = (accuracy + 1.96**2 / (2 * n)) / denominator
    margin = 1.96 * math.sqrt((accuracy * (1 - accuracy) + 1.96**2 / (4 * n)) / n) / denominator
    return {"n": n, "correct": correct, "accuracy": accuracy, "coverage": coverage,
            "failures": failures, "abstentions": abstentions,
            "wilson_95_ci": [max(0.0, centre - margin), min(1.0, centre + margin)],
            "meets_95": accuracy >= 0.95, "accuracy_ge_0_95": accuracy >= 0.95}


def verify_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Verify manifest file hashes, rejecting absolute paths and traversal."""
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    root = manifest_file.parent.resolve()
    for entry in manifest.get("files", {}).values():
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("manifest path traversal")
        target = (root / relative).resolve()
        if root not in target.parents and target != root:
            raise ValueError("manifest path traversal")
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"manifest hash mismatch: {relative}")
    return manifest
