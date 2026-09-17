"""Local ingestion, retrieval diagnostics, and separate evaluation commands."""
import argparse
import json
from pathlib import Path

from .evaluation import load_answers, load_questions, prepare_split, score_predictions, verify_manifest
from .ingest import ingest_file
from .store import SQLiteStore


def main():
    parser = argparse.ArgumentParser(prog="raglab")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--db", required=True)
    ingest.add_argument("--corpus", required=True)
    ingest.add_argument("--document", required=True)
    ingest.add_argument("--release", default="")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--db", required=True)
    search.add_argument("--corpus", action="append", required=True)
    search.add_argument("--release")
    search.add_argument("--limit", type=int, default=8)
    split = sub.add_parser("split")
    split.add_argument("dataset", type=Path)
    split.add_argument("output", type=Path)
    split.add_argument("--seed", type=int, default=20260916)
    verify = sub.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    score = sub.add_parser("score")
    score.add_argument("questions", type=Path)
    score.add_argument("answers", type=Path)
    score.add_argument("predictions", type=Path)
    args = parser.parse_args()
    if args.command == "ingest":
        chunks = ingest_file(args.path, corpus_id=args.corpus, document_id=args.document, release=args.release)
        with SQLiteStore(args.db) as store:
            store.replace_document(args.corpus, args.document, chunks)
            result = {"chunks": len(chunks), "corpus": args.corpus, "fingerprint": store.fingerprint([args.corpus])}
    elif args.command == "search":
        with SQLiteStore(args.db) as store:
            result = [e.to_dict() for e in store.search(args.query, corpus_ids=args.corpus, release=args.release, limit=args.limit)]
    elif args.command == "split":
        result = prepare_split(args.dataset, args.output, seed=args.seed)
    elif args.command == "verify":
        result = verify_manifest(args.manifest)
    else:
        result = score_predictions(load_questions(args.questions), load_answers(args.answers), args.predictions)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
