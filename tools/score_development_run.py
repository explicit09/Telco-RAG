"""Score a completed frozen development run; never interpret it as goal success."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'extensions/rag-lab'))
from raglab.evaluation import load_questions,load_answers,score_predictions


def score_run(predictions,questions,answers):
    predictions,questions,answers=map(Path,(predictions,questions,answers))
    if questions.name != 'dev.questions.jsonl' or answers.name != 'dev.answers.jsonl' or questions.parent.resolve() != answers.parent.resolve():
        raise ValueError('paired development split files are required')
    split_manifest = json.loads((questions.parent / 'manifest.json').read_text())
    for source in (questions, answers):
        entry = split_manifest['files'][source.name]
        if entry['path'] != source.name or hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('development split checksum mismatch')
    manifest=json.loads(predictions.with_suffix('.manifest.json').read_text())
    if manifest.get('status')!='completed' or manifest.get('development_only') is not True:
        raise ValueError('only completed development runs can be scored here')
    if hashlib.sha256(predictions.read_bytes()).hexdigest()!=manifest.get('predictions_sha256'):
        raise ValueError('prediction checksum mismatch')
    if hashlib.sha256(questions.read_bytes()).hexdigest()!=manifest.get('question_file_sha256'):
        raise ValueError('question-file checksum mismatch')
    all_questions={q.id:q for q in load_questions(questions)}
    identifiers=manifest['question_ids']
    if not identifiers or len(set(identifiers))!=len(identifiers) or set(identifiers)-set(all_questions):
        raise ValueError('invalid planned question IDs')
    selected=[all_questions[identifier] for identifier in identifiers]
    gold=load_answers(answers)
    if set(identifiers)-set(gold):
        raise ValueError('missing reference answers')
    score=score_predictions(selected,{identifier:gold[identifier] for identifier in identifiers},predictions)
    return {'split':'development','goal_achieved':False,'model':manifest['model'],
            'planned_questions':len(identifiers),'score':score,
            'predictions_sha256':manifest['predictions_sha256'],
            'answers_sha256':hashlib.sha256(answers.read_bytes()).hexdigest(),
            'interpretation':'Development sample only. No held-out accuracy or goal completion is established.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('predictions',type=Path)
    p.add_argument('questions',type=Path)
    p.add_argument('answers',type=Path)
    p.add_argument('report',type=Path)
    a=p.parse_args()
    result=score_run(a.predictions,a.questions,a.answers)
    with a.report.open('x') as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
