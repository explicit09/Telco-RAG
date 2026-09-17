"""Compare completed, matched development runs; retain every planned question."""
import argparse
import json
from pathlib import Path
from score_development_run import score_run
from raglab.evaluation import load_answers


def compare(left, right, questions, answers):
    left, right = Path(left), Path(right)
    scores = [score_run(path, questions, answers) for path in (left, right)]
    manifests = [json.loads(path.with_suffix('.manifest.json').read_text()) for path in (left, right)]
    for field in ('model', 'top_k', 'codex_version', 'os_denied_roots', 'question_ids',
                  'question_file_sha256', 'code_sha256'):
        if manifests[0].get(field) != manifests[1].get(field):
            raise ValueError(f'unmatched comparison: {field}')
    # Indexing duration changes after a verified resume; it is not corpus content.
    provenance = [{key:{k:v for k,v in value.items() if k != 'seconds'}
                   for key,value in manifest['corpus_manifest'].items()} for manifest in manifests]
    if provenance[0] != provenance[1]:
        raise ValueError('unmatched comparison: corpus provenance')
    predictions = [json.loads(path.read_text()) for path in (left, right)]
    gold = load_answers(answers)
    outcomes = {'both_correct': [], 'right_improved': [], 'right_regressed': [], 'both_incorrect': []}
    for identifier in manifests[0]['question_ids']:
        correct = []
        for prediction in predictions:
            row = prediction.get(identifier, {})
            correct.append(not row.get('failed') and not row.get('abstained') and row.get('selected_option') == gold[identifier])
        category = ('both_correct' if all(correct) else 'right_improved' if correct[1]
                    else 'right_regressed' if correct[0] else 'both_incorrect')
        outcomes[category].append(identifier)
    return {'split':'development', 'goal_achieved':False,
            'left':scores[0], 'right':scores[1],
            'accuracy_difference':scores[1]['score']['accuracy']-scores[0]['score']['accuracy'],
            'paired_counts':{key:len(value) for key,value in outcomes.items()},
            'paired_question_ids':outcomes,
            'interpretation':'Matched development comparison only; no held-out result or causal attribution to reranking is established.'}


def main():
    parser = argparse.ArgumentParser()
    for name in ('left','right','questions','answers','output'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    report = compare(args.left,args.right,args.questions,args.answers)
    with args.output.open('x') as handle:
        json.dump(report,handle,indent=2)
    print(json.dumps({key:value for key,value in report.items() if key not in ('left','right','paired_question_ids')},indent=2))


if __name__ == '__main__':
    main()
