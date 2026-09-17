"""Bounded subscription-backed development runs through the fork's Query flow.

Accepts only a development question file. Gold scoring is a separate command.
"""
import argparse
import hashlib
import shutil
import subprocess
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'Telco-RAG_api'), str(ROOT / 'extensions/rag-lab')]
from src.query import Query
from src.corpus import StoreCorpus
from raglab.models import Question, Evidence
from raglab.evaluation import load_questions
from raglab.store import SQLiteStore
from raglab.registry import CorpusRegistry
from raglab.retrieval import Retriever
from raglab.subscription import SubscriptionGenerator
from raglab.followup import answer_with_followup


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def resume_predictions(output, manifest_path, expected):
    existing = json.loads(manifest_path.read_text())
    for key, value in expected.items():
        if key != 'status' and existing.get(key) != value:
            raise ValueError(f'cannot resume: configuration changed at {key}')
    predictions = json.loads(output.read_text()) if output.exists() else {}
    if not isinstance(predictions, dict) or set(predictions) - set(expected['question_ids']):
        raise ValueError('invalid resumed prediction IDs')
    if output.exists() and existing.get('predictions_sha256') != hashlib.sha256(output.read_bytes()).hexdigest():
        raise ValueError('resumed predictions do not match checkpoint hash')
    return predictions, existing.get('provider_requests', 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('questions', type=Path)
    p.add_argument('database', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--limit', type=int, default=3)
    p.add_argument('--followups', type=int, choices=(0, 1, 2), default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--model', required=True)
    p.add_argument('--codex', default='codex')
    p.add_argument('--deny-read-root', action='append', type=Path, default=[])
    p.add_argument('--top-k', type=int, default=8)
    p.add_argument('--reranker', type=Path)
    p.add_argument('--phrase-search', action='store_true')
    p.add_argument('--document-reads', action='store_true', help='Allow bounded same-section reads within the follow-up budget')
    p.add_argument('--rerank-strategy', choices=('replace', 'interleave'), default='replace')
    a = p.parse_args()
    if a.questions.name != 'dev.questions.jsonl':
        raise ValueError('Only dev.questions.jsonl is accepted; this runner is not held-out isolated')
    if not 1 <= a.limit <= 100 or not 1 <= a.top_k <= 12:
        raise ValueError('development limits: 1..100 questions, 1..12 passages')
    if a.output.exists() and not a.resume:
        raise FileExistsError('refusing to overwrite run')
    questions = load_questions(a.questions)[:a.limit]
    reranker = None
    if a.reranker:
        from raglab.reranking import ONNXReranker
        reranker = ONNXReranker(a.reranker)
    predictions = {}
    generator = SubscriptionGenerator(model=a.model, max_requests=(2 + a.followups) * len(questions), executable=a.codex, deny_read_roots=a.deny_read_root, allow_document_reads=a.document_reads)
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
        'development_only': True, 'model': a.model, 'top_k': a.top_k, 'followups': a.followups,
        'codex_executable': shutil.which(a.codex),
        'os_denied_roots': [str(path.resolve()) for path in a.deny_read_root],
        'codex_version': subprocess.run([a.codex, '--version'], capture_output=True, text=True, check=True).stdout.strip(),
        'question_ids': [q.id for q in questions],
        'question_file_sha256': hashlib.sha256(a.questions.read_bytes()).hexdigest(),
        'corpus_manifest': provenance,
        'code_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files},
        'status': 'running',
        'reranker': reranker.provenance if reranker else None,
        'phrase_search': a.phrase_search,
        'document_reads': a.document_reads,
        'rerank_strategy': a.rerank_strategy,
    }
    manifest_path = a.output.with_suffix('.manifest.json')
    previous_requests = 0
    if a.resume:
        predictions, previous_requests = resume_predictions(a.output, manifest_path, run_manifest)
    elif manifest_path.exists():
        raise FileExistsError('refusing to overwrite run manifest')
    else:
        archive = a.output.with_suffix('.sources.zip')
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
            for source in code_files:
                bundle.write(source, str(source.relative_to(ROOT)))
    run_manifest['provider_requests'] = previous_requests
    if a.output.exists():
        run_manifest['predictions_sha256'] = hashlib.sha256(a.output.read_bytes()).hexdigest()
    atomic_json(manifest_path, run_manifest)
    with store_provider as store:
        search_provider = Retriever(store, reranker=reranker, phrase_search=a.phrase_search, rerank_strategy=a.rerank_strategy) if reranker or a.phrase_search else store
        for question in questions:
            if question.id in predictions:
                continue
            start = time.monotonic()
            flow = Query(question.text, [], corpus=StoreCorpus(search_provider, corpus_ids=question.corpus_ids, release=question.release))

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
                answer = answer_with_followup(question, evidence, generator, search_provider, limit=a.top_k,
                                              max_followups=a.followups, initial_query=flow.enhanced_query,
                                              allow_document_reads=a.document_reads)
                predictions[question.id] = answer.to_dict()
            except Exception as exc:
                predictions[question.id] = {'question_id': question.id, 'selected_option': None, 'abstained': False, 'failed': True, 'error': str(exc), 'development_only': True}
            predictions[question.id]['seconds'] = round(time.monotonic() - start, 2)
            atomic_json(a.output, predictions)
            run_manifest.update(predictions_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(), provider_requests=previous_requests+generator.requests)
            atomic_json(manifest_path, run_manifest)
            print(f'completed {len(predictions)}/{len(questions)} development questions', flush=True)
    run_manifest.update(status='completed', predictions_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(), provider_requests=previous_requests+generator.requests)
    atomic_json(manifest_path, run_manifest)


if __name__ == '__main__':
    main()
