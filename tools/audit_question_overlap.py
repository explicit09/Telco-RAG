"""Question-only lexical overlap screen. Does not open answer files."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def tokens(text):
    text=re.sub(r'\[\s*3GPP Release \d+\s*\]','',text,flags=re.I)
    return set(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('split',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--threshold',type=float,default=.85)
    a=p.parse_args()
    if not 0<a.threshold<=1:
        raise ValueError('threshold must be in (0,1]')
    paths={s:a.split/f'{s}.questions.jsonl' for s in ('dev','test')}
    rows={s:[json.loads(line) for line in path.read_text().splitlines()] for s,path in paths.items()}
    test=[(q,tokens(q['text'])) for q in rows['test']]
    pairs=[]
    for dev in rows['dev']:
        left=tokens(dev['text'])
        for question,right in test:
            score=len(left&right)/len(left|right) if left|right else 0
            if score>=a.threshold:
                pairs.append({'dev_id':dev['id'],'test_id':question['id'],'token_jaccard':score,'same_release':dev.get('release')==question.get('release')})
    report={'method':f'Question-only word-set Jaccard >= {a.threshold} after removing release annotation; candidates, not certified semantic duplicates',
            'pairs':pairs,'pair_count':len(pairs),'gold_read':False,
            'source_sha256':{s:hashlib.sha256(path.read_bytes()).hexdigest() for s,path in paths.items()}}
    with a.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps({'pair_count':len(pairs),'gold_read':False}))


if __name__=='__main__':
    main()
