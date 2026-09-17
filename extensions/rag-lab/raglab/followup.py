"""Bounded, evidence-only follow-up answering shared across corpora."""
from dataclasses import replace
from .generation import validate_read_request


def answer_with_followup(question, evidence, generator, store, *, limit=8, max_followups=1, initial_query=None, allow_document_reads=False):
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
    reads = 0
    seen_reads = set()
    for step in range(max_followups + 1):
        answer = generator.generate(question, evidence)
        rounds.append({'query':query,'evidence':[item.to_dict() for item in evidence], 'answer':answer.to_dict()})
        query = answer.trace.get('next_search')
        request = answer.trace.get('next_read')
        if not answer.abstained or step == max_followups:
            break
        if request is not None:
            if not allow_document_reads or query is not None:
                raise ValueError('document reads disabled or combined with search')
            validate_read_request(request, evidence)
            key = (request['anchor_id'], request['before'], request['after'])
            if key in seen_reads:
                break
            seen_reads.add(key)
            fresh = list(store.read_window(request['anchor_id'], corpus_ids=question.corpus_ids,
                         release=question.release, before=request['before'], after=request['after'],
                         limit=min(16, 2*limit), max_chars=24000))
            reads += 1
            query = 'read:' + repr(key)
        else:
            if not isinstance(query, str) or not query.strip():
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
    return replace(answer, trace={**answer.trace,'followup_searches':searches,'followup_reads':reads,'retrieval_rounds':rounds})
