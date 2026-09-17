"""Prepare a text-defined 3GPP-only TeleQnA split without printing questions/gold.

Run with `uv run --with pyzipper python tools/prepare_teleqna.py ARCHIVE OUTPUT`.
Requires the pinned archive. OUTPUT is private local evaluation data, never a corpus.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extensions/rag-lab"))
from raglab.evaluation import prepare_split, verify_manifest

ARCHIVE_SHA256 = "5302f1df9e0e66aafd12b945af518211bd41aa77294b0cac05e4234eaf12b0a5"
SOURCE_COMMIT = "d7ee5e3a705240075ecbc87cce01f1f85db5d947"


def requested_release(question):
    releases = set(re.findall(r"\b3GPP\s+Release\s+(\d+)\b", question, re.I))
    if len(releases) != 1:
        raise ValueError("question must identify exactly one 3GPP release")
    return next(iter(releases))


def main():
    import pyzipper
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if hashlib.sha256(args.archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("archive does not match pinned source")
    if args.output.exists():
        raise FileExistsError("refusing to replace benchmark output")
    with pyzipper.AESZipFile(args.archive) as archive:
        data = json.loads(archive.read("TeleQnA.txt", pwd=b"teleqnadataset"))
    reference_code = "\n".join(p.read_text(errors="replace") for p in (ROOT / "Telco-RAG_api").rglob("*.py"))
    selected, excluded_exposed = [], []
    eligible = 0
    for identifier, row in data.items():
        question = row["question"]
        # Eligibility uses only category and question, never answers/explanations.
        if row["category"] != "Standards specifications":
            continue
        if not re.search(r"\b3GPP\b", question, re.I) or re.search(r"\bIEEE\b", question, re.I):
            continue
        eligible += 1
        if identifier == "question 2045" or question in reference_code:
            excluded_exposed.append(identifier)
            continue
        options = {k: v for k, v in row.items() if re.fullmatch(r"option \d+", k)}
        answer = row["answer"].split(":", 1)[0].strip()
        if answer not in options:
            raise ValueError("unrecognized answer format")
        release = requested_release(question)
        selected.append(dict(id=identifier, text=question, options=options, answer=answer,
                             corpus_ids=["3gpp-r" + release], release=release))
    args.output.mkdir(parents=True)
    source = args.output / "normalized.private.jsonl"
    source.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    manifest = prepare_split(source, args.output / "split")
    verify_manifest(args.output / "split/manifest.json")
    protocol = {
        "source_repository": "netop-team/TeleQnA", "source_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "eligibility": "Standards specifications; question contains whole word 3GPP and no whole word IEEE",
        "eligible_before_exposure_exclusions": eligible,
        "excluded_exposed_ids": excluded_exposed,
        "counts": manifest["counts"],
        "release_policy": "Preserve the release stated in the public question; retrieve only its matching corpus and release.",
        "limitations": ["Text-defined subset, not the paper's unpublished subset manifest.",
                        "Exact normalized duplicate questions are grouped; semantic near-duplicates are not certified absent.",
                        "Public benchmark pretraining exposure cannot be ruled out.",
                        "Local gold files require process isolation before held-out generation."]
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
