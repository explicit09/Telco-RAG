# Telco-RAG improvements

This is a fork of netop-team/Telco-RAG, based on commit
`df92a34d3ad824fe4ef45b81fc956498a353a943`. The original commit remains the
unmodified code baseline. No baseline accuracy has been reproduced yet.

## Current integration

The original `Query.get_3GPP_context()` flow remains the default. It now loads
newly routed series in the second pass, keeps candidate context as complete
strings, and discards FAISS's -1 missing-neighbor indices.

An optional `corpus` parameter on `Query` replaces the 3GPP-specific router and
FAISS retrieval with a provider's `search(query, limit=...)` method. The original
candidate-generation and validation stages are reused. `src.corpus.StoreCorpus`
adapts the companion search store with explicit corpus and release filters.
A provider may wrap another database without changing Query. Two small synthetic
corpora exercise this connection; this is not a real cross-domain accuracy result.

The `extensions/rag-lab` package provides supporting ingestion and evaluation
utilities. Its separate experimental RAG pipeline is not the fork's production
query path. The legacy routed path still requires its original dependencies and model credentials.
The provider path now loads without those clients and accepts a completion callback.

## Local diagnostics

From the repository root:

```sh
python3 -m unittest discover -s tests -v
cd extensions/rag-lab
python3 -m unittest discover -s tests -v
python3 -m raglab.cli ingest manual.docx --db corpus.sqlite --corpus equipment --document manual
python3 -m raglab.cli search 'timer expiry' --db corpus.sqlite --corpus equipment
```

Supported ingestion formats are DOCX, Markdown, text, and JSONL. DOCX tables are
represented as row text; complex merged-cell semantics are not fully handled.
Lexical search works without model calls. Dense retrieval/reranking interfaces
exist, but no concrete embedding model has yet been connected or benchmarked.

## Evaluation protocol

Target: at least 95% accuracy on a fixed held-out 3GPP standards subset, with
all missing/failed/abstained questions counted in the denominator. Achieving this
score is unproven. It is not a promise of 95% on every future database.

`tools/prepare_teleqna.py` verifies the pinned TeleQnA archive SHA256, selects
Standards specifications questions containing whole-word 3GPP but not IEEE,
excludes known exposed examples, groups normalized exact duplicates, and writes
separate development/test question and answer files with SHA256 manifests.
The first split has 1,056 development and 452 test questions (seed 20260916).
Eligibility uses no answers or model performance. This is a text-defined subset,
not a reproduction of the paper's unpublished question-ID subset.

Keep generated benchmark files outside this repository and out of retrieval
indexes. Do not inspect held-out errors to tune the system. The current split
has not been certified free of semantic near-duplicates, and public benchmark
pretraining exposure cannot be ruled out. No held-out generation has run.

Subscription-backed Codex trials are intended for development. CLI read-only
mode does not prevent reading local answer files; strict held-out runs require
a separate process environment without access to gold data. No paid API calls
are authorized or have been made. The optional OpenAI API generator defaults to
refusing paid calls until explicitly enabled.

## Real-corpus development tools

- `tools/download_corpus.py`: downloads DOCX files at the pinned 3GPP corpus revision and checks upstream Git/LFS hashes.
- `tools/index_corpus.py`: checks SHA256 manifests, indexes each document transactionally, and records resumable source/parser fingerprints.
- `tools/run_development.py`: accepts a development question file, runs the original Query candidate prompt and two retrieval passes, then produces a cited structured answer via the Codex subscription. It rejects held-out filenames and never loads an answer file. Filename gating is a misuse guard, not OS isolation.

The HTTP RFC 9110 corpus has passed one real two-pass smoke question with a matching supporting quotation. This is a functional portability check, not an accuracy benchmark. The first subscription trials used the CLI default model and are exploratory. The development runner now requires an explicit `--model` and writes a manifest containing code hashes, question IDs, corpus provenance, request count, and prediction checksum. Calls use no API key, but consume subscription usage.

## Release metadata correction

Benchmark preparation v1 incorrectly assigned Release 18 to all questions.
Preparation v2 preserves the release explicitly stated in each public question
and selects a matching corpus ID. Exact question IDs, split membership, wording,
options, and gold files were verified unchanged. No held-out inference preceded
this correction. The first three-question v1 diagnostic is superseded and must
not be presented as a valid release-matched benchmark score.

The development partition requests Releases 14, 16, 17, 18, and 19. All five
releases are now indexed: 6,439 documents and 5,833,722 chunks. Database row and
distinct-document counts match the index manifests, whose source-manifest hashes
were verified. This establishes index completeness against the pinned inventories,
not semantic fidelity of mirror parsing or answerability of every question.
Requests must not be silently dropped, relabeled, or answered from a different
release without a separately justified cross-version protocol. Direct downloads
of two official ETSI Release 17 sources returned HTTP 403; the GSMA mirror supplied
the missing releases instead.

## Multi-release mirror and registry

The GSMA/3GPP mirror is pinned to
`a056f6018a7e8e67052aa68a702e272d0ae95d75`. Paginated inventories contain
1,269 Release 14, 1,453 Release 16, 1,607 Release 17, and 1,557 Release 19
Markdown documents. Original standards remain subject to their owners' terms;
source documents and generated indexes are local research data outside this repository.

`tools/sync_missing_releases.py WORKSPACE` resumes downloads and indexing, verifies
source hashes, and stops before another release if less than 6 GiB is free.
A JSON corpus registry maps each corpus ID to a separate SQLite path. Pass that
registry in place of the database argument to `tools/run_development.py`.
Requests route only to the selected corpus and release, with no implicit fallback.
Full original-baseline reproduction and held-out OS/process isolation remain
outstanding. No 95% result has been established.

## Current development execution

The shell CLI 0.147.0 cannot use the app's configured `gpt-6-astra` model.
The verified bundled CLI is `/Applications/ChatGPT.app/Contents/Resources/codex`
(version 0.154.0-alpha.6.2); pass it with `--codex`. The release-matched lexical
run of the first three development questions answered all three correctly. This
is a smoke result, not a 95% benchmark claim or an original-paper reproduction.

`--reranker DIRECTORY` enables the local ONNX cross-encoder. The current model
is `cross-encoder/ms-marco-MiniLM-L6-v2` at revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`, using `onnx/model_qint8_arm64.onnx`
and `tokenizer.json`. Install the optional runtime using
`extensions/rag-lab/requirements-rerank.txt`. Each run records model/tokenizer
hashes and runtime versions. It reranks 40 lexical candidates; there is no dense
index yet. It cannot recover evidence absent from that candidate set.

`--deny-read-root DIRECTORY` wraps the actual CLI in a macOS sandbox denying
reads and writes to that directory. Synthetic direct/symlink/hardlink checks
were denied, and the HTTP answering smoke test succeeded with the entire local
work directory blocked. This is a tested development safeguard; the runner still
refuses held-out mode. Broader isolation and contamination audits remain pending.

Runs archive their source code and checkpoint predictions after each question.
`--resume` accepts only matching configuration and prediction hashes, and skips
completed questions. An interruption between prediction and manifest writes
fails closed and requires checkpoint reconciliation rather than silent reuse.

## Question-overlap screen

A question-only lexical screen (word-set Jaccard at least 0.85, excluding release
annotations) flagged two cross-release dev/test pairs. The development members,
`question 4440` and `question 5341`, had not appeared in any model run and were
quarantined from future development. Protocol v3 has 1,054 development questions
and the same 452 held-out questions. Held-out question and answer files are
byte-for-byte identical to v2. The ongoing first-20 development run is unaffected.

`tools/audit_question_overlap.py` performs the question-only screen;
`tools/quarantine_overlap.py` refuses to quarantine an already-used development
candidate and preserves held-out files. The repeated lexical screen found zero
remaining cross-split candidates. This is not proof that every semantic
near-duplicate or pretraining exposure is absent.

## Development scoring and performance checks

`tools/score_development_run.py PREDICTIONS DEV_QUESTIONS DEV_ANSWERS REPORT`
requires a completed development manifest and verifies prediction, question, and
reference-answer hashes. Every planned question remains in the denominator.
`tools/compare_development_runs.py LEFT RIGHT DEV_QUESTIONS DEV_ANSWERS REPORT`
checks matching model, code, question IDs, corpus provenance, and execution settings,
then reports paired improvements/regressions. Indexing durations are excluded from
the corpus comparison. These reports explicitly do not establish goal achievement.

`tools/prototype_rank_stream.py` preserves the initial optimization experiment.
The tested streaming implementation is now integrated into the store. It streams SQLite's ranked matches, includes all
ties at the cutoff, and sorts tied IDs deterministically. Synthetic scope/tie tests
and one development query from each of the five releases returned exactly the same
top 40 IDs and scores as the existing search. Observed streaming times were about
2.4–5.9 seconds versus 11.7–100.8 seconds for the exhaustive query. These are small,
uncontrolled timing diagnostics with possible cache and concurrent-job effects,
not a general performance guarantee. The completed lexical and reranked comparisons
used the original search implementation; their archived sources remain available.
Quoted phrases are now preserved as phrases rather than split into OR terms.

## Completed first-20 comparison and corpus coverage repair

On the same 20 development questions with `gpt-6-astra`, the lexical baseline
scored 9/20 (45%, eight abstentions) and the reranked variant scored 12/20
(60%, five abstentions). Neither had execution failures. Nine questions were
correct in both, three improved, and none regressed. This small development
comparison does not establish a held-out result or causal proof of a reranking gain.

The parsed GSMA inventories omit some original specifications, including Release
17 TS 22.261 and TS 23.501. Inventory completeness is therefore not standards
coverage completeness. `tools/inventory_original_gaps.py` compares original
filenames against parsed document directories or an existing DOCX manifest.
An all-release audit identified omitted originals; download and ingestion of those
files is underway. Existing baseline indexes are preserved for comparison.

`tools/freeze_development_sample.py` selects 100 previously unused development
questions by seeded ID hashes, reserves at least one per release, and copies
labels only after selection. The frozen sample contains 51 Release 18, 44 Release
17, three Release 14, one Release 16, and one Release 19 question. It remains a
development sample; the separate 452-question held-out set is unchanged.

## Supplemental databases

A registry entry may be a single database path or a list, for example
`{"3gpp-r17": ["3gpp-r17.sqlite", "3gpp-r17-supplement.sqlite"]}`. Searches
apply the requested corpus and release to every part, then fuse the rankings.
Each part retains its own provenance manifest. This permits adding the omitted
original standards without modifying the databases used by the first baselines.
Duplicate database paths are rejected; conflicting chunk IDs remain errors.

## Bounded follow-up retrieval

The development runner accepts `--followups 0|1|2` (default 0) and up to 100
explicitly requested development questions. After an abstention with a proposed
search, it can retrieve fresh evidence within the same corpus/release and retry.
Repeated queries and searches yielding no new evidence stop without another
model call. Every answering round records its evidence and answer. Request limits
include the original candidate-generation call and all allowed follow-ups.

The original-file audit initially overcounted split filenames and non-numbered
attachments as missing specifications. Canonical specification IDs now group
split sections and identify already-covered specifications. The supplemental
manifest lists every exclusion and its source-only reason. Release 17's 259
candidate original files reduce to 111 files for 57 missing numbered specifications;
148 attachments or already-covered files are excluded. A malformed non-numbered
attachment stopped the first attempt; that incomplete index is not registered.
This corpus correction does not alter benchmark membership or reference answers.
