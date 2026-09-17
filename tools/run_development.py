"""Bounded subscription-backed development runs through the fork's Query flow.

Accepts only a development question file. Gold scoring is a separate command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'Telco-RAG_api'), str(ROOT / 'extensions/rag-lab')]
from src.query import Query
from src.corpus import StoreCorpus
from raglab.models import Question, Evidence
from raglab.evaluation import load_questions
from raglab.store import SQLiteStore
from raglab.registry import CorpusRegistry
from raglab.subscription import SubscriptionGenerator


def main():
    p = argparse.ArgumentParser()
    p.add_argument('questions', type=Path)
    p.add_argument('database', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--limit', type=int, default=3)
    p.add_argument('--model', required=True)
    p.add_argument('--top-k', type=int, default=8)
    a = p.parse_args()
    if a.questions.name != 'dev.questions.jsonl':
        raise ValueError('Only dev.questions.jsonl is accepted; this runner is not held-out isolated')
    if not 1 <= a.limit <= 20 or not 1 <= a.top_k <= 12:
        raise ValueError('development limits: 1..20 questions, 1..12 passages')
    if a.output.exists():
        raise FileExistsError('refusing to overwrite run')
    questions = load_questions(a.questions)[:a.limit]
    predictions = {}
    generator = SubscriptionGenerator(model=a.model, max_requests=2 * len(questions))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.database.suffix == '.json':
        store_provider = CorpusRegistry(a.database)
        provenance = store_provider.provenance(c for q in questions for c in q.corpus_ids)
    else:
        corpus_manifest = a.database.with_suffix('.manifest.json')
        if not corpus_manifest.is_file():
            raise ValueError('database provenance manifest is required')
        provenance = json.loads(corpus_manifest.read_text())
        store_provider = SQLiteStore(a.database)
    code_files = sorted((ROOT / 'extensions/rag-lab/raglab').glob('*.py')) + [ROOT / 'Telco-RAG_api/src/query.py', ROOT / 'Telco-RAG_api/src/corpus.py', Path(__file__)]
    run_manifest = {
        'development_only': True, 'model': a.model, 'top_k': a.top_k,
        'question_ids': [q.id for q in questions],
        'question_file_sha256': hashlib.sha256(a.questions.read_bytes()).hexdigest(),
        'corpus_manifest': provenance,
        'code_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files},
        'status': 'running',
    }
    manifest_path = a.output.with_suffix('.manifest.json')
    if manifest_path.exists():
        raise FileExistsError('refusing to overwrite run manifest')
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + '\n')
    with store_provider as store:
        for question in questions:
            start = time.monotonic()
            flow = Query(question.text, [], corpus=StoreCorpus(store, corpus_ids=question.corpus_ids, release=question.release))

            def evidence_for_flow():
                evidence = []
                for identifier in flow.context_source:
                    chunk = store.get_chunk(identifier)
                    if chunk is None or chunk.corpus_id not in question.corpus_ids or (question.release is not None and chunk.release != question.release):
                        raise ValueError('retrieval returned invalid corpus/release evidence')
                    evidence.append(Evidence(chunk, 0.0))
                return evidence

            def complete_candidate(prompt, *, model):
                candidate = Question(question.id + '-candidate', prompt, {}, question.corpus_ids, question.release)
                response = generator.generate(candidate, evidence_for_flow())
                return 'NO' if response.abstained else response.text

            flow.completion = complete_candidate
            try:
                # The fork's original candidate-generation prompt and two-pass flow.
                flow.get_3GPP_context(k=a.top_k, validate_flag=False)
                evidence = evidence_for_flow()
                if not evidence:
                    raise ValueError('no evidence retrieved')
                answer = generator.generate(question, evidence)
                predictions[question.id] = answer.to_dict()
            except Exception as exc:
                predictions[question.id] = {'question_id': question.id, 'selected_option': None, 'abstained': False, 'failed': True, 'error': str(exc), 'development_only': True}
            predictions[question.id]['seconds'] = round(time.monotonic() - start, 2)
            a.output.write_text(json.dumps(predictions, indent=2) + '\n')
            print(f'completed {len(predictions)}/{len(questions)} development questions', flush=True)
    run_manifest.update(status='completed', predictions_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(), provider_requests=generator.requests)
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
