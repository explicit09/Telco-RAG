"""Quarantine unused development overlap candidates; never change held-out files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path)
    p.add_argument('audit',type=Path)
    p.add_argument('run_directory',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    audit=json.loads(a.audit.read_text())
    excluded={pair['dev_id'] for pair in audit['pairs']}
    used=set()
    for path in a.run_directory.glob('*.manifest.json'):
        used.update(json.loads(path.read_text()).get('question_ids',[]))
    for path in a.run_directory.glob('*.json'):
        record=json.loads(path.read_text())
        if isinstance(record,dict):
            used.update(key for key,value in record.items() if isinstance(value,dict) and 'selected_option' in value)
    if excluded & used:
        raise ValueError('an overlap candidate has already been used by a model run; manual protocol review required')
    if a.output.exists():
        raise FileExistsError('refusing to overwrite protocol version')
    dev=[json.loads(x) for x in (a.source/'dev.questions.jsonl').read_text().splitlines()]
    if not excluded <= {q['id'] for q in dev}:
        raise ValueError('audit refers to unknown development IDs')
    a.output.mkdir(parents=True)
    for name in ('test.questions.jsonl','test.answers.jsonl'):
        shutil.copyfile(a.source/name,a.output/name)
        if (a.source/name).read_bytes()!=(a.output/name).read_bytes():
            raise ValueError('held-out bytes changed')
    for name,id_key in (('dev.questions.jsonl','id'),('dev.answers.jsonl','question_id')):
        lines=(a.source/name).read_text().splitlines()
        kept=[line for line in lines if json.loads(line)[id_key] not in excluded]
        (a.output/name).write_text('\n'.join(kept)+'\n')
    files={}
    for name in ('dev.questions.jsonl','dev.answers.jsonl','test.questions.jsonl','test.answers.jsonl'):
        path=a.output/name
        files[name]={'path':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'count':len(path.read_text().splitlines())}
    manifest={'version':3,'source_manifest_sha256':hashlib.sha256((a.source/'manifest.json').read_bytes()).hexdigest(),
              'quarantined_development_ids':sorted(excluded),'held_out_bytes_unchanged':True,'files':files,
              'counts':{'dev':files['dev.questions.jsonl']['count'],'test':files['test.questions.jsonl']['count']},
              'rationale':audit['method'],'gold_exposed_to_model':False}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({key:manifest[key] for key in ('version','quarantined_development_ids','held_out_bytes_unchanged','counts')},indent=2))


if __name__=='__main__':
    main()
