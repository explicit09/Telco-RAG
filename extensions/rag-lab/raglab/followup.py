"""Bounded, evidence-only follow-up answering shared across corpora."""
from dataclasses import replace


def answer_with_followup(question, evidence, generator, store, *, limit=8, max_followups=1, initial_query=None):
    if not 0 <= max_followups <= 2 or limit < 1:
        raise ValueError('invalid follow-up limits')

    def validate(items):
        for item in items:
            chunk = item.chunk
            if chunk.corpus_id not in question.corpus_ids or (question.release is not None and chunk.release != question.release):
                raise ValueError('follow-up evidence violates corpus/release scope')

    evidence = list(evidence)
    validate(evidence)
    rounds = []
    seen_queries = {' '.join(question.text.casefold().split())}
    query = initial_query or question.text
    searches = 0
    for step in range(max_followups + 1):
        answer = generator.generate(question, evidence)
        rounds.append({'query':query,'evidence':[item.to_dict() for item in evidence], 'answer':answer.to_dict()})
        query = answer.trace.get('next_search')
        if not answer.abstained or step == max_followups or not isinstance(query, str) or not query.strip():
            break
        normalized = ' '.join(query.casefold().split())
        if normalized in seen_queries:
            break
        seen_queries.add(normalized)
        fresh = list(store.search(query, corpus_ids=question.corpus_ids, release=question.release, limit=limit))
        searches += 1
        validate(fresh)
        previous = {item.chunk.id:item.chunk for item in evidence}
        if any(item.chunk.id in previous and previous[item.chunk.id] != item.chunk for item in fresh):
            raise ValueError('conflicting follow-up chunk content')
        if not any(item.chunk.id not in previous for item in fresh):
            break
        combined = {}
        for item in fresh + evidence:
            if item.chunk.id in combined and combined[item.chunk.id].chunk != item.chunk:
                raise ValueError('conflicting follow-up chunk content')
            combined.setdefault(item.chunk.id,item)
        evidence = list(combined.values())[:2 * limit]
    return replace(answer, trace={**answer.trace,'followup_searches':searches,'retrieval_rounds':rounds})
