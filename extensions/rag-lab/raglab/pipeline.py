from __future__ import annotations

import time
from dataclasses import replace

from .models import Answer, Generator, Question
from .retrieval import Retriever, fuse_rankings


class RAG:
    def __init__(self, retriever: Retriever, generator: Generator, *, max_rounds: int = 2,
                 top_k: int = 8, max_context_chars: int = 16000):
        if not 1 <= max_rounds <= 5 or top_k < 1 or max_context_chars < 256:
            raise ValueError("Invalid pipeline budget")
        self.retriever, self.generator = retriever, generator
        self.max_rounds, self.top_k, self.max_context_chars = max_rounds, top_k, max_context_chars

    def answer(self, question: Question) -> Answer:
        if not question.corpus_ids or not question.text.strip():
            raise ValueError("Questions require nonempty text and explicit corpora")
        started = time.monotonic()
        query, seen_queries, rankings, steps = question.text, set(), [], []
        last = Answer(question.id, "No evidence found.", None, (), True)
        previous_ids = set()
        for round_number in range(1, self.max_rounds + 1):
            if query.casefold().strip() in seen_queries:
                break
            seen_queries.add(query.casefold().strip())
            results = self.retriever.search(query, corpus_ids=question.corpus_ids, release=question.release,
                                           limit=self.top_k, candidates=max(40, self.top_k))
            rankings.append(results)
            merged = fuse_rankings(rankings)[:self.top_k]
            ids = {e.chunk.id for e in merged}
            if not ids or (round_number > 1 and ids == previous_ids):
                steps.append({"round": round_number, "query": query, "stop": "no_new_evidence"})
                break
            previous_ids = ids
            evidence, remaining = [], self.max_context_chars
            for item in merged:
                if remaining <= 0:
                    break
                text = item.chunk.text[:remaining]
                evidence.append(replace(item, chunk=replace(item.chunk, text=text)))
                remaining -= len(text)
            steps.append({"round": round_number, "query": query, "chunk_ids": [e.chunk.id for e in evidence]})
            last = self.generator.generate(question, evidence)
            if last.question_id != question.id:
                raise ValueError("Generator returned an answer for another question")
            if any(c not in {e.chunk.id for e in evidence} for c in last.citations):
                raise ValueError("Generator cited evidence it was not given")
            next_query = last.trace.get("next_search")
            if not last.abstained or not isinstance(next_query, str) or not next_query.strip():
                break
            query = next_query.strip()
        return replace(last, trace={**last.trace, "retrieval_steps": steps,
                                   "elapsed_seconds": time.monotonic() - started,
                                   "max_rounds": self.max_rounds})
