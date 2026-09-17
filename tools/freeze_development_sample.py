"""Freeze an unused, release-covered development sample without inspecting labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def select(questions, excluded, count, seed):
    if len({q['id'] for q in questions}) != len(questions):
        raise ValueError('duplicate question IDs')
    eligible = [q for q in questions if q['id'] not in excluded]
    if not 1 <= count <= len(eligible):
        raise ValueError('invalid sample size')
    key = lambda q: (hashlib.sha256(f"{seed}:{q['id']}".encode()).hexdigest(), q['id'])
    ordered = sorted(eligible, key=key)
    representatives = {}
    for q in ordered:
        representatives.setdefault(q['release'], q)
    if count < len(representatives):
        raise ValueError('sample cannot cover every release')
    chosen = {q['id'] for q in representatives.values()}
    for q in ordered:
        if len(chosen) == count:
            break
        chosen.add(q['id'])
    return [q for q in ordered if q['id'] in chosen]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('split',type=Path)
    p.add_argument('runs',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--count',type=int,default=100)
    p.add_argument('--seed',type=int,default=20260916)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('sample already exists')
    manifest = json.loads((a.split/'manifest.json').read_text())
    sources = {}
    for name in ('dev.questions.jsonl','dev.answers.jsonl'):
        sources[name] = (a.split/name).read_bytes()
        if hashlib.sha256(sources[name]).hexdigest() != manifest['files'][name]['sha256']:
            raise ValueError('source split checksum mismatch')
    excluded = set()
    for path in a.runs.glob('*.manifest.json'):
        excluded.update(json.loads(path.read_text()).get('question_ids',[]))
    for path in a.runs.glob('*.json'):
        record = json.loads(path.read_text())
        if isinstance(record,dict):
            excluded.update(k for k,v in record.items() if isinstance(v,dict) and 'selected_option' in v)
    questions = [json.loads(line) for line in sources['dev.questions.jsonl'].splitlines()]
    chosen = select(questions,excluded,a.count,a.seed)
    # Selection has already finished; now copy the corresponding labels privately.
    gold = {row['question_id']:row for row in map(json.loads,sources['dev.answers.jsonl'].splitlines())}
    if any(q['id'] not in gold for q in chosen):
        raise ValueError('missing development answer')
    a.output.mkdir(parents=True)
    files = {}
    for name, rows in [('dev.questions.jsonl',chosen),('dev.answers.jsonl',[gold[q['id']] for q in chosen])]:
        path = a.output/name
        path.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows))
        files[name] = {'path':name,'count':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    result = {'split':'development','seed':a.seed,'count':len(chosen),'files':files,
              'release_counts':dict(Counter(q['release'] for q in chosen)),
              'excluded_previously_planned_ids':sorted(excluded),
              'source_manifest_sha256':hashlib.sha256((a.split/'manifest.json').read_bytes()).hexdigest(),
              'selection':'SHA256(seed:ID) order, reserve one unused question per release, then fill to count; no labels or outcomes used.'}
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('split','count','release_counts','selection')},indent=2))


if __name__ == '__main__':
    main()
