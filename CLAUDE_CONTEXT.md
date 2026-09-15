# RecallBench — Hand-Built Vector Search, Honestly Benchmarked Against FAISS

---

## What This Project Is
A from-scratch implementation of HNSW (Hierarchical Navigable Small World graphs)
— the algorithm that powers real vector databases like Pinecone and Weaviate —
rigorously benchmarked against FAISS (the industry-standard library) on recall
and latency, then wrapped in a working RAG (Retrieval-Augmented Generation)
pipeline so it does real work, not just sit in a benchmark script.

---

## Why We Are Building This
- Every other AI-adjacent resume says "built a RAG app with LangChain + Pinecone"
  — that's assembling components, not proving you understand what's inside them
- Implementing HNSW yourself (graph construction + greedy multi-layer search) is
  real algorithms/data-structures work, not an API wrapper
- The benchmark against FAISS is the actual differentiator, not the algorithm
  alone — a documented, honest comparison ("here's where mine loses and why")
  survives interview follow-ups; "I built a vector search" alone does not
- Produces a concrete numbers table for the resume: recall@k, p50/p95 latency,
  build time, memory — all self-measured, not claimed
- Pairs with existing portfolio: PromptGate already proves LLM-evaluation
  judgment, BuildBoard already proved rigorous benchmarking discipline (17-test
  suite + real p95 latency numbers found a real production bug) — RecallBench
  extends that same discipline into algorithms/IR territory PromptGate doesn't
  cover
- Can be demoed live: query box → retrieved chunks → generated answer →
  side-by-side comparison against FAISS on the same query

---

## The Core Tradeoff This Project Is About
Exact nearest-neighbor search (checking every vector) is accurate but slow —
O(n) per query. HNSW trades a small amount of accuracy for massive speed by
building a multi-layer graph that lets a search skip most of the dataset per
query. This project's entire intellectual content is implementing that
tradeoff correctly and then **measuring** exactly how much accuracy was traded
for how much speed — never assumed, always benchmarked.

---

## How We Work — The Development Process
This project is built in **batches**, the same way BuildBoard was. Each batch
is one focused, self-contained unit of work (one algorithm piece, one
integration, one feature) — never multiple unrelated things bundled together.

The process for every single batch, in order, with no steps skipped:

1. **Explain first.** Before any code is written, Claude explains what the
   batch will build, why it's needed, what approach will be used, and what
   alternatives were considered and rejected (and why). This includes every
   design decision, not just the big ones — a decision as small as "why this
   variable is named this way" doesn't need a debate, but any decision that
   affects correctness, performance, or the eventual benchmark numbers does.
2. **Build it**, with every line explained as it's written (per The Rules
   above).
3. **Test it against what we actually set out to do for that batch** — not
   just "does it run without crashing," but "does it do the specific thing
   this batch was supposed to do, verified concretely" (a real query result,
   a real measured number, a real comparison — the same standard BuildBoard's
   batches were held to, e.g. Batch 4's real webhook round-trip test, not a
   simulated one).
4. **Log it** — see Key Decisions Log below. This happens before moving to
   the next batch, not retroactively at the end.

### Why we log decisions and bugs this thoroughly
Every decision made, every bug found, and exactly how it was diagnosed and
fixed gets written into the Key Decisions Log below, in the same depth as
BuildBoard's log. This is not busywork — it is the raw material for two
things that matter for interviews and the resume:
- A **README** written afterward from these logs (same as BuildBoard's
  README "Engineering Log" section, which came directly from its
  CLAUDE_CONTEXT.md decision log) — so the finished project has a real,
  specific engineering narrative instead of a generic feature list.
- **Interview answers** — "tell me about a bug you found" or "walk me
  through a design decision" should be answerable by pointing at a real
  logged entry, not reconstructed from memory months later.

A bug entry in the log must include: what broke, how it was discovered
(ideally via a test or benchmark, not by inspection), the root cause (not
just the symptom), and the fix. This mirrors the real bug BuildBoard's
benchmark caught (the Decimal/float crash) — that entry is a template for
the level of detail expected here.

---

## My Setup
- OS: macOS (same machine as BuildBoard)
- Reuse what's already installed/known-working from BuildBoard where sensible
  (Python via pyenv, Docker Desktop, GitHub CLI already set up)
- This project is Python-only on the backend (no Java/Spring here — that's
  FinFlow's lane) — keep frontend to React + TypeScript, consistent with
  BuildBoard/PromptGate/FinFlow

---

## Tech Stack
- Core algorithm:      Python 3.11, NumPy only (NO ann libraries — hnswlib,
                        annoy, etc. are explicitly banned from the core
                        implementation; using one would defeat the entire
                        point of this project)
- Comparison baseline: FAISS (`faiss-cpu` via pip) — the industry reference
                        point we benchmark against, never replace
- Embeddings:          sentence-transformers, `all-MiniLM-L6-v2` (small, fast,
                        free — no per-query API cost for embedding the corpus)
- LLM (RAG answer step): Claude API — Haiku for dev, matching BuildBoard/
                        PromptGate's existing pattern; AI_MOCK flag for CI,
                        same convention as those two projects
- Backend:             FastAPI (async) — same framework as BuildBoard, no
                        reason to introduce a new one
- Database (if needed for storing benchmark run history): PostgreSQL, same
                        pattern as BuildBoard/PromptGate
- Frontend:            React + TypeScript + Vite — consistent with the rest
                        of the portfolio
- Corpus:              Wikipedia paragraph embeddings or similar precomputed
                        dataset (~100k+ vectors) — avoid re-embedding a huge
                        corpus from scratch if a suitable precomputed one
                        exists, to save time and (if using a paid embedding
                        API instead of sentence-transformers) cost

---

## Documentation & Git Hygiene Standards

**README and all docs** are written in first-person ownership voice — every
decision, tradeoff, and bug fix is presented as something you engineered,
because it is: every batch requires you to understand and approve each line
before it's accepted (per The Rules below), so the decisions documented are
genuinely yours, not a transcription of what a tool produced unsupervised.
No meta-commentary about AI tooling, no "written with the help of..."
framing anywhere in the polished docs — the engineering log entries
described above are written the same way BuildBoard's and PromptGate's
existing READMEs are: as an engineer's own account of what was built and why.

**Git commits** are logically scoped (one coherent change per commit, not a
grab-bag), with descriptive messages explaining the *why* behind a change,
not just restating the diff. No "wip"/"fix"/"fix again" chains — squash or
organize before pushing so the history itself reads as a clean, deliberate
build, the same discipline BuildBoard's commit history follows.

Note on commit trailers: no `Co-Authored-By` (or any other AI-attribution)
trailer in this project's commits — explicitly overridden by direct user
instruction when the repository was first pushed. (An earlier version of
this note claimed this trailer couldn't be turned off; that was wrong — a
direct user instruction in-session overrides the tool's default
attribution behavior.)

---

## THE RULES — Read Before Every Session

### For Claude In VS Code / Claude Code:
1. Before writing ANY code — explain what it does and why
2. Explain EVERY single line of code written, especially in the HNSW core
   (graph construction and search) — this is the part that must be deeply
   understood, not just working
3. After each batch — tell me exactly what to run to verify
4. Wait for my confirmation before starting the next batch
5. If I ask WHY — stop everything and explain fully
6. Never assume I know something — explain from first principles
7. If something fails — explain WHY it failed before fixing it
8. Always check this file before doing anything
9. NEVER skip or fake a benchmark number. If something can't be measured yet,
   say so explicitly rather than estimating or guessing a plausible-looking
   number — every metric in the final resume bullet must trace back to an
   actual measured run, the same standard BuildBoard's benchmark held to

### For Me:
1. Never accept code I do not understand
2. Ask WHY for every line that is unclear, especially in the HNSW algorithm
3. Verify each batch works before saying confirmed
4. Update this file's Key Decisions Log after every batch
5. Start a new session when Claude starts forgetting context (see below)

---

## When To Start A New Claude Session

Same triggers as BuildBoard:
```
1. Claude gives wrong answers about decisions made earlier
2. Claude starts regenerating files that already exist
3. Claude contradicts itself within the same session
4. The session has gone longer than 3-4 batches
5. You get a confusing error Claude cannot explain
6. Claude stops following the explanation rules
```

### How To Start A New Session Correctly:

Paste this EXACTLY at the start of every new session:

```
Read CLAUDE_CONTEXT.md completely before doing anything.

We are building RecallBench — a hand-built HNSW vector search implementation,
benchmarked against FAISS, wrapped in a RAG pipeline.

Completed batches so far: [LIST THEM]
Current batch to work on: Batch [X] — [BATCH NAME]

Key decisions already made:
[COPY FROM KEY DECISIONS SECTION BELOW]

Rules reminder:
- Explain everything before coding it
- Explain every single line, especially the HNSW core algorithm
- Wait for my confirmation between batches
- If I ask why — stop and explain fully
- Never fake or estimate a benchmark number — measure it for real or say
  it hasn't been measured yet

Begin Batch [X] now.
```

---

## The Metrics That MUST Be Measured (This Is The Resume Payload)

Every one of these must come from an actual executed benchmark run, never
estimated. This table is the actual deliverable of the whole project:

| Metric | Your HNSW | FAISS |
|---|---|---|
| Recall@10 (vs. brute-force ground truth) | measure | measure (near 100%, reference) |
| Query latency p50 | measure | measure |
| Query latency p95 | measure | measure |
| Index build time | measure | measure |
| Memory footprint | measure | measure |

Do not consider the benchmark batch complete until every cell in this table
has a real number from a real run, on a real corpus of realistic size
(~100k+ vectors — a toy 100-vector test does not count as the reported
number, only as a correctness sanity check).

---

## Planned Batch Roadmap
(Adjust as we go — update this section as batches complete or the plan changes)

```
Batch 1:  HNSW core algorithm — graph construction + node insertion [DONE]
Batch 2:  HNSW search — greedy multi-layer traversal [DONE]
Batch 3:  Brute-force exact search (ground truth for recall computation) [DONE]
Batch 4:  Corpus loading + embedding pipeline (sentence-transformers) [DONE]
Batch 5:  Benchmark harness — recall@k, latency percentiles, build time, memory [DONE]
Batch 6:  FAISS integration — the comparison side of the benchmark [DONE]
Batch 7:  RAG pipeline — retrieval -> prompt assembly -> Claude answer [DONE, live Claude call unverified — no API key in this env]
Batch 8:  Backend API (FastAPI) wrapping the pipeline + benchmark endpoints [DONE]
Batch 9:  Frontend demo UI — query box, retrieved chunks, answer, live
          comparison table [DONE]
Batch 10: Deployment (mirror BuildBoard's AWS pattern, if going live)
```

---

## Key Decisions Log
UPDATE THIS AFTER EVERY BATCH, BEFORE STARTING THE NEXT ONE — same discipline
as BuildBoard's log. Each batch entry should cover, in this order:

1. What was built
2. What was discussed/decided along the way, and why (including alternatives
   considered and rejected)
3. How it was tested against the batch's actual goal (the specific command
   run, the specific real output/number observed — not "it works")
4. Any bug found: what broke, how it was discovered, the root cause, the fix
5. Anything explicitly deferred to a later batch, and why

This log is the source material for the eventual README — write each entry
as if it will be read later without any other context, the same way
BuildBoard's log entries were detailed enough to turn directly into its
README's Engineering Log section.

```
### Batch 1: HNSW core algorithm — graph construction + node insertion

**What was built**
- `core/hnsw.py`: `HNSW` class with `insert(node_id, vector)` as the public
  entry point, plus internals: `_random_level` (exponential level draw),
  `_search_layer` (bounded best-first beam search within one layer, used
  internally by insertion — not the public query API), `_select_neighbors`
  (naive top-M), `_add_edge`/`_prune_connections` (bidirectional wiring +
  degree-cap enforcement).
- `tests/test_construction.py`: structural correctness test suite (not a
  recall/perf benchmark — that's Batch 5) run against a real 2000x50 random
  index.
- `requirements.txt` (numpy only), `.venv` (Python 3.11.9 via pyenv, matching
  the rest of the portfolio), `.gitignore`.

**Decisions made, and why**
- Distance metric: squared L2 (`np.dot(diff, diff)`), not cosine, and not
  plain (non-squared) L2. Sqrt is monotonic and wasted work for ranking.
  Once Batch 4 uses L2-normalized sentence-transformer embeddings, L2²
  ranking is mathematically identical to cosine ranking, so this costs
  nothing later. Rejected: implementing cosine distance directly — more
  code, no benefit given normalized vectors will be used.
- Neighbor selection: naive top-M nearest (`_select_neighbors`), not the
  HNSW paper's diversity-aware heuristic (SELECT-NEIGHBORS-HEURISTIC).
  Deliberately deferred rather than baked in up front, so that if the
  Batch 5/6 recall benchmark against FAISS comes back worse than expected,
  there's a real, measured, attributable reason to point at and fix — not
  a guessed optimization applied preemptively.
- Parameters: M=16, M0=32 (2xM), ef_construction=200, mL=1/ln(M) — all from
  the original Malkov & Yashunin paper's recommended defaults, not
  arbitrary. M0 is doubled because layer 0 carries all fine-grained
  connectivity responsibility.
- `_search_layer`'s early-termination condition (stop expanding once the
  closest remaining candidate is worse than the current worst kept result)
  is a heuristic bound, not a proof of correctness — this is standard for
  HNSW and is why `ef_construction` is set generously (200) to compensate.
- Used a private `random.Random(seed)` instance (seed=42) rather than the
  global `random` module, so index construction is reproducible regardless
  of what else in the process calls `random.*` — required for the
  benchmark numbers to be re-runnable and get the same graph.
- Edges are enforced bidirectional everywhere (`_add_edge` and
  `_prune_connections` both maintain both directions) — a graph-theoretic
  requirement for HNSW; a one-way edge would silently break reachability
  for anything approaching from the "wrong" side.

**How it was tested against this batch's actual goal**
Ran `tests/test_construction.py` (2000 random 50-dim vectors, M=16,
ef_construction=200, seed=42). All checks passed on first run:
- All 2000 nodes present at layer 0.
- All edges bidirectional across all layers (45,614 directed edges
  checked).
- No node exceeds its layer's degree cap (M0=32 at layer 0, M=16 above).
- Layer-membership consistency holds (no node present at a layer above its
  recorded top layer, and present at every layer below it).
- BFS connectivity from the entry point through the layer-0 graph reaches
  all 2000 nodes (the check that most directly predicts whether Batch 2's
  search will be able to find everything).
- Layer sizes: [2000, 109, 7], max_level=2 (expected ~log_M(n)=2.74).
  Shrink ratios 0.054 and 0.064 vs. target ~0.0625 (1/M) — matches
  expected exponential layer decay.
- Informal sanity check: node 0's layer-0 graph neighbors overlap its true
  10-NN by 7/10 (not a real recall number — that's Batch 5 — just a smoke
  test that the graph isn't producing nonsense).

**Bugs found**
None — all structural checks passed on the first run.

**Deferred to later batches**
- Diversity-aware neighbor selection heuristic (only add if the real
  recall benchmark in Batch 5/6 shows naive top-M underperforming FAISS
  meaningfully).
- `_search_layer` is only exercised here as an internal insertion helper;
  Batch 2 builds the actual public multi-layer query API (`ef`-tunable
  top-k search) on top of it.
- Dimension validation on `insert()` — no current caller passes mismatched
  dimensions, so not added; revisit if that changes.

### Batch 2: HNSW search — greedy multi-layer traversal

**What was built**
- `core/hnsw.py`: added `search(query, k, ef=None)`, the public k-NN query
  API. Refactored the greedy upper-layer descent (previously inlined in
  `insert()`) into a shared `_greedy_descend(query, ep, from_layer,
  to_layer)`, used by both `insert()` and `search()` — pure refactor,
  confirmed behavior-identical by re-running the Batch 1 suite before and
  after (byte-identical output).
- `tests/test_search.py`: correctness + real recall-sweep test suite.

**Decisions made, and why**
- `ef` (layer-0 beam width) is a separate, per-query, caller-tunable
  parameter — not reused from `ef_construction`. This is the actual
  recall/latency knob the project's whole benchmark thesis is about;
  hardcoding it would make Batch 5's ef-sweep impossible. Default when
  omitted: `max(k, 50)` — never smaller than k, 50 as a reasonable
  starting width, not a claimed recall number.
- Distances returned by `search()` stay squared L2 (not converted to true
  Euclidean via sqrt) — same reasoning as Batch 1: ranking is identical
  either way, sqrt is wasted work, and Batch 5's recall computation only
  needs correct ID sets/ordering, never distance magnitude.
- Return shape is `list[(node_id, distance)]`, ascending by distance,
  length <= k — flipped from `_search_layer`'s internal `(distance, id)`
  convention for a more natural public-facing API.
- Extracted `_greedy_descend` rather than leaving the descent logic
  duplicated between `insert()` and `search()` — both perform the exact
  same "cheap ef=1 walk down through sparse upper layers" step; keeping it
  in one place means a future correction only has to be made once.

**How it was tested against this batch's actual goal**
Ran `tests/test_search.py` on a 2000x50 random index (M=16,
ef_construction=200, seed=42). All checks passed on first run:
- Empty index returns `[]`, no exception.
- Querying with a vector already in the index returns that exact node
  first, at distance ~0 (2.00e-16 scale) — strong signal the graph stays
  self-reachable.
- `k=10` search returns exactly 10 unique node ids, sorted ascending by
  distance.
- `k=100` against a 5-node index returns all 5 available nodes, no crash.
- **Real recall@10 measured** over 200 held-out random queries (brute-force
  ground truth computed independently via `np.argsort`, not reusing any
  HNSW code), swept across ef:
  - ef=1   -> recall@10 = 0.766
  - ef=10  -> recall@10 = 0.766
  - ef=50  -> recall@10 = 0.982
  - ef=200 -> recall@10 = 1.000
  Recall is monotonically non-decreasing as ef increases and converges to
  1.0 — the actual recall/latency tradeoff curve this project exists to
  characterize, demonstrated on a real (if small, non-benchmark-scale)
  dataset. This is not the formal benchmark deliverable (Batch 5, with
  FAISS side-by-side and the ~100k-vector corpus) — just proof search is
  correct and behaves as the algorithm predicts.

**Bugs found**
None — all checks passed on first run.

**Investigated after initial review (closed, not left open)**
`ef=1` and `ef=10` were originally reported as producing identical recall
(0.766) with only a hand-wave guess ("plausible on this data
distribution"). Went back and actually diagnosed it rather than leaving
that as a guess:
- Confirmed with `_search_layer` called directly (bypassing `search()`)
  that it genuinely returns different, correctly-sized candidate sets at
  different `ef` values (1 vs 5 vs 10 vs 50 all differ) — so the beam
  search itself was never the issue.
- Root cause: `search()`'s line `self._search_layer(query, ep, max(ef, k),
  layer=0)` clamps the effective beam width to `max(ef, k)`. With k=10,
  every requested `ef` from 1 to 10 gets silently upgraded to an effective
  ef=10 — so they were *never* going to differ. Confirmed by sweeping
  `ef=1,2,5,9,10,11,15,50`: recall is flat at 0.766 for every ef <= 10,
  then starts moving immediately at ef=11 (0.784) and climbs from there
  (ef=15 -> 0.850, ef=50 -> 0.982).
- This is the clamp working exactly as designed (see Batch 2's own
  decision log: "ef can never sensibly be smaller than k") — not a bug,
  and not a coincidence of the data distribution as first guessed. The
  original explanation was wrong; this is the corrected, verified one.

**Deferred to later batches**
- Formal recall@k / latency percentile / build-time / memory benchmark
  harness, and the FAISS side-by-side comparison — Batch 5/6, on the real
  ~100k+ vector corpus, not this batch's small synthetic dataset.
- Brute-force exact search as a standalone reusable module (Batch 3) —
  this batch's ground truth was computed inline in the test script only,
  good enough for this batch's correctness check but not intended to be
  the reusable ground-truth implementation the benchmark harness will use.

### Batch 3: Brute-force exact search (ground truth for recall computation)

**What was built**
- `core/brute_force.py`: `BruteForce` class — exact k-NN by checking every
  vector, vectorized with NumPy (einsum for distance computation,
  argpartition + partial sort for top-k selection instead of a full
  argsort over the whole corpus). Deliberately independent of any HNSW
  internals — this is the oracle HNSW gets graded against, so it can't
  share HNSW's bugs.
- `tests/test_brute_force.py`: correctness suite, including a
  hand-checkable example and cross-checks against both an independently
  computed NumPy ground truth and HNSW's own distance function.

**Decisions made, and why**
- Same distance metric as HNSW: squared L2, kept textually identical
  on purpose — recall is only meaningful when both sides rank by the same
  metric, so this isn't a preference, it's a correctness requirement.
- Bulk `__init__(vectors, ids)` instead of mirroring HNSW's one-at-a-time
  `insert()`. Rejected the mirrored-API approach: brute force has no graph
  to build incrementally, and the benchmark harness will always load the
  whole corpus once before running queries — an incremental insert API
  here would be complexity with no real use.
- Explicit `ids` array (not assumed `0..n-1`) so ground truth can be
  computed over the exact same id space as the HNSW index being graded,
  even with non-integer ids (e.g. string document ids from the real
  corpus later).
- `np.argpartition` + partial sort instead of a full `np.argsort` over the
  whole corpus for every query — O(n) selection instead of O(n log n),
  meaningful at ~100k+ vectors since brute force is the intentionally-slow
  baseline and shouldn't be made needlessly slower on top of that.
- No batched multi-query method — deferred. The benchmark needs per-query
  latency (p50/p95), so queries get timed individually regardless; batching
  the distance computation wouldn't serve that measurement.

**How it was tested against this batch's actual goal**
Ran `tests/test_brute_force.py`. All 7 checks passed after one fix:
- Hand-checkable 1D example (points at 0,1,2,5,10; query at 3) — verified
  the returned order and distances match values computed by hand.
- Custom (non-numeric) ids are respected and returned correctly.
- `k` larger than corpus returns all available vectors, no crash.
- Returned ids/distances are plain python `int`/`float`, not numpy scalar
  types — matters for JSON serialization once this is exposed via the API
  in Batch 8.
- Results sorted ascending by distance.
- **Regression check**: `BruteForce.search` matches an independently
  computed full-`argsort` ground truth exactly (same id set, same
  distances to 1e-9) on 3000 vectors, dim=30, k=15 — proves the
  argpartition optimization didn't introduce an error.
- **Cross-check**: `BruteForce` and `HNSW._distance_to_query` compute the
  literal same squared-L2 distance for the same pair of vectors
  (32.176658 both sides) — confirms the two modules that will be compared
  in Batch 5 actually share a distance definition.

**Bug found**
- **What broke**: `check_ids_are_respected` crashed with
  `ValueError: invalid literal for int() with base 10: np.str_('doc_a')`.
- **How it was discovered**: a test that deliberately passed string ids
  (`["doc_a", "doc_b", "doc_c"]`) to `BruteForce`, to check the arbitrary-id
  design decision above actually works — not found by inspection.
- **Root cause**: `search()`'s return line used `int(self.ids[i])` to
  convert out of numpy's scalar type, but `int()` only works when the
  underlying value is actually numeric — it assumed ids would always be
  integer-like, contradicting the module's own design decision to support
  arbitrary ids.
- **Fix**: replaced `int(self.ids[i])` with `self.ids[i].item()`, which
  converts any numpy scalar back to its correct native python type
  (`np.str_` -> `str`, `np.int64` -> `int`) generically, rather than
  assuming one specific type.
- **Second issue caught during the same fix cycle**: after fixing the code,
  the test itself still failed — `check_ids_are_respected`'s query
  `[0.5, 0.5]` was exactly equidistant (both 0.5) from two of the three
  fixture points, an unintentional tie in the test fixture, not a bug in
  `BruteForce`. Fixed by changing the query to `[0.9, 0.9]`, which is
  unambiguously closest to one point.

**Gaps closed after initial review (same batch, before moving on)**
Two open questions were raised after the initial pass and closed before
confirming this batch, rather than carried forward:
- **Tie-breaking was previously undefined.** `argpartition`/`argsort`
  don't guarantee any particular order among exactly-tied distances, which
  meant `BruteForce.search` could return a different (but equally valid)
  answer on different runs/numpy versions for tied inputs — a real
  reproducibility gap for something meant to be *the* ground truth. Fixed
  by ranking the top-k candidates with `np.lexsort((ids, dists))` —
  distance primary, id ascending as the tiebreaker — so results are fully
  deterministic. Verified with a new test using 4 points forming two exact
  distance ties (with ids deliberately out of numeric/insertion order, to
  rule out the tie-break "working" by accident): confirmed ties resolve by
  ascending id, and confirmed via a second call that results are
  reproducible across repeated invocations on identical input.
- **`k=0` was untested and unhandled** — would have hit
  `np.argpartition(dists, 0)` on an already-degenerate case. Added an
  explicit early return (`if k == 0: return []`) and a test confirming it.
- Did not add a test for an empty corpus (`n=0` vectors) — ruled out of
  scope rather than silently skipped: this pipeline's `BruteForce` is
  always built from a real, non-empty corpus (the whole point is grading
  HNSW against it), so a zero-vector corpus isn't a real state this module
  needs to handle correctly, only one it happens not to crash on by luck.

**Deferred to later batches**
- Batched multi-query vectorized search — only if profiling in Batch 5
  shows per-query brute-force calls are a bottleneck for building the
  ground-truth table over many benchmark queries.
- Recall computation at the k-th boundary when the *real* corpus (Batch 4
  embeddings) contains near-duplicate/duplicate chunks close enough to
  cause near-ties — the tie-break fix above makes `BruteForce` itself
  deterministic, but Batch 5's recall metric definition still needs to
  decide whether a near-tie miss should count against HNSW; revisit with
  real data once it's available, not decided on synthetic Gaussian data
  where near-ties don't occur.

### Batch 4: Corpus loading + embedding pipeline (sentence-transformers)

**What was built**
- `core/corpus.py`: `load_wikipedia_paragraphs(n, seed, prefix_rows)` —
  loads real, paragraph-chunked Simple English Wikipedia text from a
  public HF dataset, ignoring its baked-in (Cohere) embeddings.
- `core/embed.py`: `embed_texts(texts)` — wraps `sentence-transformers`
  (`all-MiniLM-L6-v2`), returns L2-normalized float32 384-dim vectors.
- `scripts/build_corpus.py`: the real, once-run job that produces the
  actual corpus artifact used by every later batch.
- `data/corpus_embeddings.npy` (100,000 x 384 float32, 146.5MB) and
  `data/corpus_metadata.jsonl` (100,000 {id, text} rows) — real, generated
  artifacts, gitignored (regenerated via the script, not committed —
  146.5MB exceeds GitHub's 100MB no-LFS limit anyway).
- `tests/test_corpus_embed.py` (pipeline correctness on a small subset)
  and `tests/test_corpus_artifact.py` (integrity checks on the actual
  100k-row artifact that was built).

**Decisions made, and why**
- Corpus source: `timescale/wikipedia-22-12-simple-embeddings` (real,
  pre-chunked Simple English Wikipedia paragraphs), but its own Cohere
  embeddings are discarded entirely — only the `contents` text is used,
  re-embedded with our own model. Chosen over (a) raw Wikipedia + our own
  paragraph chunking (rejected: pulls Batch 7's chunking-design work
  forward into this batch) and (b) a smaller non-Wikipedia dataset
  (rejected: weaker portfolio narrative, less topical diversity) —
  discussed with and confirmed by the user before starting.
- `normalize_embeddings=True` in `embed_texts` — not just an efficiency
  knob. Batch 1 justified using squared L2 instead of cosine distance by
  claiming "identical ranking once vectors are normalized" — this flag is
  what actually makes that claim true for this corpus rather than an
  unenforced assumption. Confirmed: saved embeddings have max deviation
  from unit norm of 1.19e-07.
- Embeddings stored/compared as float32, not float64.
- Fixed-seed random sample of `n` rows from a `prefix_rows`-row prefix of
  the source file (not the full ~485k rows, not the file's first `n` rows
  in raw order) — bounds download time while avoiding a "first N rows"
  ordering bias risk (e.g. alphabetical-by-title).
- `data/` (both the raw download cache and the generated corpus artifacts)
  is gitignored; `scripts/build_corpus.py` is the source of truth an
  artifact can always be regenerated from.

**Bug found #1 — benchmark-honesty issue, found before writing corpus code**
- **What broke**: nothing crashed — this was caught by reasoning about the
  upcoming memory-footprint benchmark, not by a test. `HNSW.insert()` and
  `BruteForce.__init__` (Batches 1 and 3) force-cast every vector to
  `float64`, silently doubling memory vs. the embedding model's native
  `float32` output (100k x 384: 150MB at float32 vs. 300MB at float64).
- **Why it mattered**: memory footprint is one of the five required
  benchmark metrics in this file, and FAISS would be compared at its
  (typically float32) default — an unfair, dishonest comparison if our
  side were silently inflated to float64 for no reason.
- **Fix**: raised explicitly with the user before writing any Batch 4
  code; confirmed switching both `HNSW` and `BruteForce` to `float32`.
  Changed `dtype=np.float64` -> `dtype=np.float32` in both modules (4
  call sites total). Re-ran the full Batch 1-3 test suite afterward — all
  passed, with two test-side fixes needed for the resulting precision
  change (see next entry) — no production-code behavior changed besides
  the intended precision.
- **Downstream test fixes required by the float32 change**:
  - `test_brute_force.py`'s regression check compared `BruteForce`'s
    (now float32) output against an independently-computed float64 ground
    truth with a `1e-9` absolute tolerance — too tight for float32
    (~7 significant digits). Fixed with `np.isclose(rtol=1e-5,
    atol=1e-4)`, the mathematically appropriate comparison (scales with
    magnitude, unlike a fixed epsilon).
  - `test_brute_force.py`'s HNSW-vs-BruteForce consistency check called
    `HNSW._distance_to_query()` directly with a raw (float64) query
    array — `_distance_to_query`'s precondition (query already float32)
    is normally guaranteed by its only real callers, `search()`/
    `insert()`, which cast before calling down; the test bypassed that by
    reaching into a private method directly. Fixed the test to cast the
    query to float32 first, honoring the same precondition `search()`
    would — not a production code change.

**Bug found #2 — pandas.read_csv(url) fetching over HTTP inefficiently**
- **What broke**: nothing crashed, but `load_wikipedia_paragraphs` was
  drastically slower than it should have been — 156 seconds to fetch just
  5,000 rows.
- **How it was discovered**: noticed while validating the small-subset
  pipeline test, before committing to the real ~100k-row run (extrapolated,
  that rate would have meant ~100+ minutes for the real job) — investigated
  rather than just accepting the slow number.
- **Root cause**: measured the underlying connection directly with `curl
  --range` and got ~14.8MB/s — the network was never the bottleneck.
  `pandas.read_csv()` given a URL fetches and parses row-by-row over HTTP
  inefficiently (many small round trips), rather than streaming in large
  chunks.
- **Fix**: `core/corpus.py`'s `_download_prefix()` now downloads a
  bounded byte-range prefix (computed from the dataset's known average
  bytes/row, with a 1.3x safety margin) to a local file with `curl -L
  --range` first, then has pandas parse the *local* file — pandas over a
  local file is fast; the network fetch now actually uses the available
  bandwidth. Measured result: the same 500-paragraph load that took 156s
  dropped to 0.1s on a warm cache / ~2.4s cold for the byte-range fetch
  itself at this scale.

**How it was tested against this batch's actual goal**
- `tests/test_corpus_embed.py` on a small (500-paragraph) real subset:
  loading returns real, unique-id, non-empty text; embeddings are
  (n, 384) float32 with unit norm; a semantic sanity check (two
  same-topic sentences embed closer together than two different-topic
  ones: 0.8477 vs. 1.7387 squared-L2); and a real end-to-end check —
  built a small HNSW index from real embedded paragraphs, queried with
  the real natural-language text "science and space exploration", and
  the top-3 retrieved paragraphs were genuinely topically relevant (a
  Mars rover probe, a definition of outer space, and Eddington's 1919
  relativity expedition) — inspected by reading the actual retrieved
  text, not just a numeric pass/fail.
- **Real ~100k-scale artifact actually built**, per this batch's scope
  commitment: `scripts/build_corpus.py` loaded 100,000 real Wikipedia
  paragraphs (37.0s) and embedded them (99.4s, ~1000 texts/sec once
  batching warmed up) -> `data/corpus_embeddings.npy` (146.5MB) and
  `data/corpus_metadata.jsonl`.
- `tests/test_corpus_artifact.py` verified the actual saved artifact, not
  just the pipeline logic: correct shape (100000, 384) float32; all
  100,000 ids unique; all embeddings unit-norm (max deviation 1.19e-07);
  and critically, a row-order-consistency check — re-embedded 20 randomly
  sampled *stored* texts fresh and confirmed they matched the *stored*
  embedding at that same row exactly (max diff 0.00e+00, since model
  inference is deterministic) — this is the check that would have caught
  a silent shuffle/reorder bug between the embeddings array and the
  metadata file, which two separately-written output files are always at
  risk of.

**Deferred to later batches**
- The real recall/latency/build-time/memory benchmark against FAISS on
  this 100k corpus — Batch 5/6, not this batch. This batch only proves
  the corpus and embeddings are real, correctly ordered, and semantically
  sane.
- Whether `prefix_rows=150,000` (vs. the full ~485,859-row dataset)
  meaningfully limits corpus diversity for the benchmark — not
  investigated; revisit only if Batch 5's numbers look suspicious in a
  way that traces back to corpus composition.

### Batch 5: Benchmark harness — recall@k, latency percentiles, build time, memory

**What was built**
- `benchmarks/harness.py`: four side-agnostic measurement primitives —
  `time_build(insert_fn, ids, vectors)`, `recall_at_k(approx_ids,
  exact_ids)`, `latency_percentiles(latencies_seconds, percentiles)`, and
  `measure_peak_memory_kb()`. Deliberately index-agnostic (`time_build`
  takes a bare callable, not an index object) so Batch 6 reuses the same
  functions for FAISS's numbers instead of writing parallel measurement
  code that could silently diverge.
- `scripts/run_benchmark.py`: the real benchmark runner — loads the actual
  corpus artifact, holds out queries, builds ground truth, builds an HNSW
  index, times it, measures its memory, and sweeps `ef` measuring
  recall@10 and latency percentiles at each value. Writes results to
  `benchmarks/results/hnsw_<tag>.json`. Supports `--n` for smaller pilot
  runs before committing to the full corpus.
- `tests/test_harness.py`: correctness tests for the four harness
  functions on small, hand-checkable synthetic inputs — not the benchmark
  itself.
- `benchmarks/results/hnsw_pilot.json`, `hnsw_pilot30k.json`,
  `hnsw_full100k.json` — real, committed run outputs.

**Decisions made, and why**
- Query set: 500 vectors held out from the corpus *before* building the
  index, used as queries against the remaining index. Rejected
  self-querying (using vectors already in the index as queries) — that
  trivially returns the query itself at distance 0, inflating recall in a
  way that doesn't reflect real query behavior. We don't have real
  user-typed queries yet (that's Batch 7's RAG pipeline); held-out corpus
  paragraphs are an honest stand-in since recall@k only cares that the
  query is excluded from the index it's graded against, not where the
  query text originated. Flagged as re-measurable with real queries once
  Batch 7 exists.
- Memory measurement: process peak RSS (`resource.getrusage().ru_maxrss`)
  delta between right-after-corpus-load and right-after-build, not
  `sys.getsizeof`. `sys.getsizeof` doesn't follow references and would
  badly undercount a graph of nested dicts/sets plus NumPy's C-allocated
  buffers. `ru_maxrss` is a high-water mark, so it captures peak usage
  during the build, not just a before/after snapshot that could miss
  transient peaks. Documented explicitly that `ru_maxrss` units differ by
  platform (bytes on macOS, KB on Linux) — this codebase is macOS-only
  (see "My Setup"), so the unconditional /1024 is correct here but would
  misreport on Linux.
- `ef` sweep at {10, 50, 100, 200, 400} — the same knob Batch 2 explored
  on synthetic 2000-vector data, now measured for real. Build time and
  memory are measured once per run (they're build-time costs, independent
  of the search-time `ef` parameter); recall and latency get their own
  row per `ef` value.
- Ground truth computed once per query, before the `ef` sweep starts —
  it's `ef`-independent (exact brute-force search), so computing it 5x
  (once per sweep value) would be wasted, non-trivial work at 500
  queries x ~99.5k corpus vectors each.
- Pilot runs before the full corpus commit: ran at n=8,000 (11.7s build)
  and n=30,000 (59.2s build) first, fit the scaling exponent between them
  (~n^1.21 — build cost is mildly superlinear, since later insertions
  search a larger, denser graph than early ones), and used that to
  estimate the full 100k build at ~4-5 minutes before running it — this
  was a deliberate check before committing to an unknown-duration run,
  not a guess baked into the benchmark result itself (the actual number
  is the real 231.2s measurement below, not the extrapolation).

**How it was tested against this batch's actual goal**
- `tests/test_harness.py`: all 9 checks passed — `recall_at_k` verified
  against hand-checkable perfect/partial/no-match/empty/order-independent
  cases; `latency_percentiles` verified against a known evenly-spaced
  distribution (p50~50.5ms, p95~95.05ms expected, both matched within
  tolerance) and a single-value edge case; `time_build` verified it calls
  every (id, vector) pair in order and that elapsed time reflects real
  sleep time, not a stub; `measure_peak_memory_kb` verified it returns a
  positive number that increases after a real ~200MB allocation that
  actually touches its pages (`np.ones`, not `np.empty`, to force real
  resident memory rather than unfaulted virtual pages).
- Full pre-existing suite (Batches 1-4) re-run after adding this batch's
  code: all checks still pass, confirming nothing in `core/` was touched
  or broken.
- **Real full-corpus benchmark run** (`python scripts/run_benchmark.py
  --n 100000 --n-queries 500 --tag full100k`), on the actual 99,500-vector
  index (500 vectors held out as queries) built from the real
  `data/corpus_embeddings.npy` artifact, M=16, ef_construction=200,
  seed=42:
  - **Build time: 237.0s** (~3.9 min)
  - **Index memory: 96.6MB** (peak RSS delta, measured in the corrected
    window -- see the bug entry above; this is graph structure only --
    edge sets, dict overhead, per-node bookkeeping -- since the vectors
    themselves are stored as views into the already-loaded corpus array,
    not duplicated)
  - **Ground truth (brute-force) computation: 4.2s** for 500 queries
    against the 99,500-vector index
  - Recall@10 / latency sweep across `ef`:

    | ef | recall@10 | p50 latency | p95 latency |
    |---|---|---|---|
    | 10  | 0.8338 | 0.229ms | 0.349ms |
    | 50  | 0.9702 | 0.646ms | 0.901ms |
    | 100 | 0.9868 | 1.086ms | 1.448ms |
    | 200 | 0.9946 | 1.958ms | 2.555ms |
    | 400 | 0.9974 | 3.583ms | 4.673ms |

  - Notably, recall does *not* reach 1.000 even at ef=400 on the real
    100k corpus (unlike the 2000-vector synthetic test in Batch 2, and
    unlike the smaller pilot runs here, which did reach 1.000) — a real,
    measured effect of corpus scale, not a bug: at larger n there are more
    "near-miss" candidates competing for the top-10 boundary, so the same
    ef buys slightly less recall headroom. This is exactly the kind of
    real tradeoff number this project exists to produce, not something to
    explain away.

**Bug found — memory measurement window contaminated by an unrelated phase**
- **What broke**: nothing crashed, and the harness unit tests and full
  regression suite all passed — this was a silent correctness issue in
  the benchmark's own methodology, not caught by any test.
- **How it was discovered**: not by inspection or by a test — the user
  asked, after this batch's first pass, whether we were confident in the
  current state or should re-evaluate before moving on. Re-reading the
  script's measurement order with that question in mind surfaced it.
- **Root cause**: `measure_peak_memory_kb()`'s baseline was taken *before*
  the brute-force ground-truth computation (500 exact searches, each
  allocating a transient ~146MB (99,500 x 384 float32) diff array), not
  immediately before the HNSW build loop. `ru_maxrss` is a process-wide,
  monotonic high-water mark — it never decreases — so any transient spike
  from ground-truth's own scratch allocations, if it exceeded the HNSW
  graph's real footprint, would get counted into "index memory" anyway.
  The originally reported 227.3MB was not necessarily wrong, but it was
  unverifiably mixed with an unrelated computation's memory cost.
- **Fix**: moved the baseline measurement to immediately before
  `hnsw = HNSW(...)`, after ground truth is already fully computed, so the
  measured window contains nothing but the index build itself.
- **Verified impact of the fix**: re-ran the full 100k benchmark after the
  fix (`hnsw_full100k.json`, same seed=42, same everything else). Build
  time (237.0s vs. 231.2s) and every recall/latency number at every `ef`
  were unchanged within normal run-to-run noise, as expected -- neither
  is affected by the memory measurement code path. **Index memory dropped
  from 227.3MB to 96.6MB** -- confirming the original number really was
  inflated by roughly 130MB of ground-truth-computation contamination, not
  just theoretically at risk of it.
- **Why this matters beyond this one number**: this exact
  `measure_peak_memory_kb()` function will be reused as-is for FAISS's
  memory measurement in Batch 6 -- getting the *methodology* right here,
  not just the number, is what keeps that comparison fair.

**Deferred to later batches**
- Choosing a single "headline" `ef` value for the final resume table
  (rather than reporting the full sweep) — deferred to Batch 6, once
  FAISS's numbers exist to compare against at the same operating point.
- FAISS's build time / memory / recall / latency numbers — Batch 6, using
  these same `benchmarks/harness.py` functions so both sides are measured
  identically.
- **Vector-storage asymmetry between the two memory numbers.**
  `HNSW.insert()`'s `np.asarray(vector, dtype=np.float32)` returns a view
  into the already-loaded corpus array when the input is already float32
  (confirmed via `np.shares_memory`), not a copy -- so the 96.6MB index
  memory figure is graph structure only and does not include vector
  storage (~146MB). FAISS necessarily owns a copy of every vector
  internally, so its memory number will include that cost by
  construction. Batch 6 needs to either (a) make HNSW store owned copies
  too, for a like-for-like number representative of a real standalone
  deployed index, or (b) explicitly add the shared vector-storage cost
  back to HNSW's side before publishing the comparison table -- not
  decided yet, needs discussion at the start of Batch 6.
- Re-measuring recall/latency with genuine user-typed queries instead of
  held-out corpus paragraphs, once Batch 7's RAG pipeline exists.
- Batched/parallel query execution for the harness — not needed yet;
  500 sequential queries per ef value completes in well under a second
  of harness overhead beyond the search calls themselves.

### Batch 6: FAISS integration — the comparison side of the benchmark

**What was built**
- `core/faiss_index.py`: `FaissHNSW`, a thin wrapper around
  `faiss.IndexHNSWFlat` shaped to the exact same public interface as our
  own `HNSW` (`insert(node_id, vector)`, `search(query, k, ef=None)`, now
  also `memory_bytes()`), so `benchmarks/harness.py` and
  `scripts/run_benchmark.py` measure both index types through identical
  code with zero per-index-type branching. "Flat" means FAISS stores full
  uncompressed vectors per node, same as our own HNSW — the fair
  comparison is graph-construction quality against graph-construction
  quality, not against a compressed (PQ/SQ) FAISS variant ours was never
  asked to compete with.
- `HNSW.memory_bytes()` (`core/hnsw.py`) and `FaissHNSW.memory_bytes()`
  (`core/faiss_index.py`) — see the bug/fix entry below; these replace the
  RSS-based memory measurement entirely.
- `tests/test_faiss_index.py` — correctness suite for the wrapper
  (arbitrary ids, self-query exact match, result shape/ordering, k larger
  than index, real ef-sweep recall-convergence check) plus a
  `memory_bytes()` correctness check (deterministic, above the raw-vector
  floor, monotonic in index size).
- `tests/test_construction.py` gained `check_memory_bytes` — the same
  three properties, for our own HNSW.
- `scripts/run_benchmark.py` rewritten to build and benchmark both HNSW
  and FAISS in one script (`benchmark_one_index`, shared by both), writing
  `benchmarks/results/comparison_<tag>.json`.
- `benchmarks/results/comparison_full100k.json` — the real, final
  100k-corpus, both-index comparison result.

**Decisions made, and why**
- Distance metric inside `FaissHNSW`: `faiss.METRIC_L2` (plain, non-squared
  L2), left as FAISS's default rather than forced to match our squared-L2
  convention. Since embeddings are L2-normalized (Batch 4), rankings are
  identical either way, and recall@k only depends on rank order — matching
  Batch 1's own reasoning for using squared L2 in the first place. Not
  worth adding code to force an exact match of raw distance values that
  this project never reports or compares directly.
- `efSearch` (FAISS's ef-equivalent) is a mutable attribute on the index,
  set inside `search()` right before each call rather than once at
  construction — this is what makes the same ef-sweep this project runs
  against our own HNSW possible against FAISS too, using the same sweep
  values, in the same loop.
- `FaissHNSW` tracks its own external-id -> internal-index mapping in a
  plain Python list (`self._ids`), appended in insertion order. FAISS has
  no native concept of externally-chosen ids. This only stays correct
  because `insert()` is called in strict append order (true for every
  caller in this codebase); documented explicitly in the code as a
  precondition rather than defended against, since enforcing it generally
  would mean adding dead code no real caller needs.

**Bug found — `ru_maxrss`-based memory measurement is not reproducible,
investigated and resolved**
- **What broke**: nothing crashed. The originally-built `run_benchmark.py`
  (since replaced) measured index memory the same way Batch 5 did —
  process-wide peak RSS (`resource.getrusage().ru_maxrss`) delta,
  before/after the build. Building both HNSW and FAISS in one process
  first showed FAISS's own delta reading ~0MB (documented at the time as
  RSS already being pushed up by HNSW's build). The fix attempted for that
  — building each index in its own fresh subprocess via `subprocess.run`,
  so each process's RSS delta reflects only its own build — was
  implemented, but the underlying assumption (that an isolated process's
  RSS delta reliably reflects one index's memory) was never actually
  verified against a repeat run.
- **How it was discovered**: after VS Code lost this session's history and
  work resumed from the code + result files left on disk, the recovered
  `comparison_full100k_v2.json` showed **`our_hnsw.index_memory_mb: 0.0`**
  — inconsistent with Batch 5's own 96.6MB measurement at the same corpus
  scale. Rather than accept or explain away a 0.0, it was investigated
  directly: re-running the exact same `--only hnsw` subprocess command by
  hand gave **95.4MB**; re-running the full orchestrator (both
  subprocesses) gave **263.6MB** for the same HNSW build. Three runs of
  byte-for-byte identical code, same corpus, same seed: 0MB, 95MB, 264MB.
- **Root cause**: `ru_maxrss` is a process-wide, monotonic high-water mark
  — it reflects whatever the OS happened to reserve for the *entire*
  process's memory needs over its lifetime, not a targeted measurement of
  one data structure. Even with fresh-subprocess isolation, each
  subprocess still loads the full corpus (~146MB) and computes the
  brute-force ground truth (large transient diff-array allocations) before
  the timed build even starts — and allocator/heap-layout behavior
  (arena reuse, page allocation patterns, ambient system memory pressure
  from unrelated processes) determines how much of the graph's own
  allocation shows up as *new* peak RSS versus getting absorbed into
  memory the process already had reserved. This is a fundamentally
  unreliable signal for "how much memory does this specific index
  structure occupy," regardless of how carefully the surrounding process
  is isolated or ordered — subprocess isolation narrowed the contamination
  source but never eliminated the core problem.
- **Fix**: dropped RSS-based measurement entirely, for both index types.
  Added `HNSW.memory_bytes()` — a deterministic deep-walk of the graph's
  actual owned Python objects (`self.vectors`, `self.graph`), summing
  `sys.getsizeof` recursively through dicts/sets/lists and deduplicating
  by `id()` so shared objects (e.g. a node_id referenced as both a dict
  key and inside a neighbor set) aren't double-counted. Added
  `FaissHNSW.memory_bytes()` using `len(faiss.serialize_index(self.index))`
  — FAISS's own serialized byte size, which includes its stored vectors
  and graph structure in one deterministic number. Since `run_benchmark.py`
  no longer needs isolated processes to get an honest memory number (each
  method measures its own index directly, unaffected by anything else in
  the process), the subprocess-spawning machinery was removed entirely —
  both indices now build in one simple, sequential script, which is also
  meaningfully less code.
- **Verified the fix is actually deterministic**: added
  `check_memory_bytes` to both `tests/test_construction.py` and
  `tests/test_faiss_index.py` — each calls `memory_bytes()` twice on the
  same unmodified index and asserts bit-for-bit equality, confirms the
  result exceeds the raw vector-data floor (`n * dim * 4` bytes), and
  confirms a smaller index reports less memory than a larger one. All
  checks pass. This is the property `ru_maxrss` could never have passed —
  it gave three different answers for the identical input.
- **Resolves a Batch 5 deferred item**: Batch 5 had flagged a vector-storage
  asymmetry (our HNSW stored vectors as views into the shared corpus array,
  so its memory number excluded vector storage entirely, while FAISS would
  necessarily own copies) as needing a decision at the start of this batch.
  That decision was already made earlier in the lost session — `HNSW.insert()`
  now does `np.array(vector, dtype=np.float32, copy=True)`, so the index
  owns real copies. Combined with `memory_bytes()` now counting that owned
  `self.vectors` dict, and FAISS's serialized size inherently including its
  own stored vectors, both final numbers are like-for-like "fully deployed
  index" totals — vectors plus graph structure, nothing excluded on either
  side.

**How it was tested against this batch's actual goal**
- Full pre-existing suite (Batches 1-5) re-run after all changes: all
  checks still pass, including the two new `memory_bytes()` checks.
- **Real full-corpus comparison run**
  (`python scripts/run_benchmark.py --n 100000 --n-queries 500 --tag
  full100k`), same 99,500-vector index / 500 held-out queries split as
  Batch 5, M=16, ef_construction=200, seed=42:

  | | Our HNSW | FAISS IndexHNSWFlat |
  |---|---|---|
  | Build time | 216.4s | 123.5s |
  | Index memory | 390.0 MB | 159.4 MB |

  | ef | ours recall@10 | ours p50 | faiss recall@10 | faiss p50 |
  |---|---|---|---|---|
  | 10  | 0.8338 | 0.222ms | 0.8576 | 0.036ms |
  | 50  | 0.9702 | 0.607ms | 0.9814 | 0.108ms |
  | 100 | 0.9868 | 1.027ms | 0.9938 | 0.196ms |
  | 200 | 0.9946 | 1.835ms | 0.9986 | 0.364ms |
  | 400 | 0.9974 | 3.307ms | 0.9998 | 0.686ms |

  Raw vector data alone accounts for ~152.9MB either way (99,500 x 384
  float32). Subtracting that out: our HNSW's graph structure costs
  ~237.1MB of Python dict/set overhead, versus FAISS's ~6.5MB of packed
  C++ arrays for the equivalent graph. FAISS is also ~6x faster to search
  at every ef value and ~1.8x faster to build. None of this is surprising
  in direction — a hand-rolled Python implementation was never going to
  beat a production C++ library — but now it's a specific, honest,
  measured gap with an identified cause (Python object overhead in the
  graph representation), which is exactly the kind of finding this
  project exists to produce rather than avoid.
- Deleted `benchmarks/results/comparison_pilot.json`,
  `comparison_pilot2.json`, `comparison_pilot30k2.json`, and
  `comparison_full100k_v2.json`/`comparison_full100k_debug.json` — all
  products of the RSS-based methodology above, superseded and no longer
  trustworthy. Kept `hnsw_full100k.json`/`hnsw_pilot*.json` (Batch 5's
  HNSW-only artifacts) as historical record of that batch's own
  measurements, which are already fully documented in Batch 5's log entry
  above.

**Additional testing done before closing this batch — real cross-index
agreement on real corpus text**
Everything above was either synthetic-data correctness (unit tests) or an
aggregate statistic (recall@10 averaged over 500 queries). Neither is the
same as actually reading what comes back, the way Batch 4's
`check_end_to_end_real_query` did for our own HNSW alone. Added
`tests/test_cross_index_retrieval.py`: builds BruteForce, our HNSW, and
FaissHNSW from the same real 10,000-vector corpus subset, issues 5 real
natural-language queries ("science and space exploration", "history of
ancient Rome", "programming languages and computers", "music genres and
instruments", "climate change and the environment"), and prints each
method's top-5 retrieved paragraph text side by side for direct
inspection, plus computes overlap between all three.

Result: **100% agreement** across all three retrieval paths on all 5
queries at k=5, ef=100 — every method returned the exact same 5 paragraph
ids for every query, and every retrieved paragraph is genuinely topical
(e.g. the space-exploration query returned Mars colonization and the Space
Race; the ancient-Rome query returned Vespasian's reign and Roman
polytheism). This is the first time FAISS's real output was read by eye
rather than only measured statistically, and it's also the first time our
HNSW, FAISS, and exact ground truth were compared side by side on the same
real query rather than only against each other's aggregate numbers. Full
output preserved in this log entry's test run; assertions require >=0.7
recall against ground truth for both approximate methods as a floor (both
scored 1.0 in the actual run).

**Deferred to later batches**
- FAISS's build-time advantage isn't broken down further (e.g. how much is
  its multithreaded C++ implementation vs. algorithmic differences) — not
  needed for this project's core deliverable, which is the recall/latency/
  memory comparison, not a FAISS internals investigation.
- Choosing a single "headline" ef value for the resume table — deferred to
  README-writing time, once all batches are done and the full narrative is
  visible.
- This project currently has no git repository (lost along with session
  history in the VS Code update, or never initialized) despite
  `CLAUDE_CONTEXT.md`'s git hygiene section assuming one — flagged to the
  user, not resolved in this batch; needs a decision on whether to `git
  init` fresh or whether a remote already exists somewhere to reconnect to.
```

### Batch 7: RAG pipeline — retrieval -> prompt assembly -> Claude answer

**What was built**
- `core/config.py`: `Settings` (pydantic-settings), matching BuildBoard's
  `claude_service.py`/`config.py` convention exactly — `anthropic_api_key`,
  `ai_mock` (defaults `True`), `claude_model` (defaults
  `claude-haiku-4-5-20251001`), loaded from `.env`.
- `.env.example` (committed) and `.env` (gitignored) — same split as
  BuildBoard.
- `core/rag.py`: `retrieve()` (runs our own HNSW's `search()` and attaches
  real corpus text), `build_prompt()` (numbered context passages + the
  question), `generate_answer()` (real Claude call or `[MOCK]` response
  depending on `settings.ai_mock`), `answer_query()` (top-level: embed ->
  retrieve -> prompt -> answer).
- `tests/test_rag.py`: hand-checkable prompt-assembly test, mock-answer
  labeling test, and a real end-to-end retrieval test against a real
  3,000-vector corpus subset.
- `requirements.txt` gained `anthropic`, `pydantic-settings`.

**Decisions made, and why**
- Retrieval uses **our own hand-built HNSW**, not FAISS. FAISS was only
  ever the benchmark's comparison baseline (Batch 6) — the entire point of
  this project is the hand-built implementation doing real retrieval work,
  so the RAG pipeline exists to prove that, not to route around it.
- Grounded prompt design: the system prompt explicitly instructs Claude to
  answer *only* from the numbered context passages, say so when the
  context doesn't contain the answer, and cite which passage(s) it used.
  Rejected a looser prompt that let Claude blend in its own general
  knowledge — that would make it impossible to tell, from the answer
  alone, whether retrieval actually worked or Claude just answered from
  training data. A grounded prompt turns "does retrieval work" into
  something checkable by reading the answer.
- `AI_MOCK` defaults to `True` (not `False`) — same reasoning as
  BuildBoard: a fresh checkout must never be able to bill the Claude API
  by accident just by running the test suite. The mock response is
  explicitly labeled `[MOCK]` and still includes the real retrieved
  passage ids, so the retrieval half of the pipeline is genuinely
  exercised even when the API call itself is stubbed.
- `embed_fn` is passed into `answer_query()` as a parameter rather than
  `core.rag` importing `core.embed` directly — same "inject the callable"
  pattern `benchmarks/harness.py`'s `time_build` already uses for
  `insert_fn`. Lets tests that only care about prompt assembly or the
  `AI_MOCK` path avoid loading the real sentence-transformers model.
- Test corpus subset: 3,000 real vectors, not the full 100k. This is a
  logic/correctness test for the RAG plumbing, not a benchmark — a fresh
  100k HNSW build takes ~3.5 minutes, far too slow to re-run repeatedly
  for something that only needs to prove retrieval + prompt assembly work
  on real text.

**How it was tested against this batch's actual goal**
- `check_build_prompt_hand_checkable`: fixed, fake input; confirms both
  numbered passages and the question text land in the assembled prompt.
- `check_mock_answer_is_labeled_and_uses_real_ids`: confirms the mock path
  is clearly labeled AND actually reflects the real retrieved ids passed
  in, not a hardcoded placeholder blind to its input.
- `check_retrieve_and_answer_query_end_to_end_real_corpus`: real corpus
  text, real `sentence-transformers` embedding, real HNSW retrieval.
  Query "What is the greenhouse effect?" against a real 3,000-paragraph
  subset returned real, topically relevant passages (fossil-fuel carbon
  emissions, photosynthesis/carbon exchange) in ascending-distance order,
  and `answer_query()`'s full return shape (`query`/`retrieved`/`answer`)
  was verified end to end.
- `check_live_claude_call_if_key_available`: written to make one real,
  live Claude call and verify a real (non-mock) answer comes back, **but
  it did not run** — there is no `ANTHROPIC_API_KEY` available in this
  development environment. The check explicitly prints that it's skipped
  and why, rather than silently passing or faking a result — per rule 9,
  an untested code path is reported as untested, not claimed as verified.
  This is the one piece of Batch 7 without real-world confirmation yet.

**Bugs found**
None — all runnable checks passed on the first run.

**Deferred to later batches**
- Verifying the real (non-mock) Claude API call actually works — needs a
  real `ANTHROPIC_API_KEY` in `.env`, which isn't available in this
  environment. Revisit as soon as one is provided; until then, the live
  path is implemented per BuildBoard's proven pattern but unverified.
- Chunk size/overlap strategy: this batch retrieves whole corpus
  paragraphs as-is (they're already reasonably sized, pre-chunked by the
  dataset itself — see Batch 4) rather than re-chunking them. Revisit only
  if real answers in Batch 8/9 usage show passages are too long/short for
  good grounding.
- Persisting a built HNSW index to disk (pickle or similar) instead of
  rebuilding on every process start — not needed yet since nothing in this
  batch is a long-running server; becomes necessary once Batch 8's FastAPI
  backend needs to serve queries without a multi-minute startup cost.
- Wiring this pipeline into an API/UI — Batches 8 and 9.
```

### Batch 8: Backend API (FastAPI) wrapping the pipeline + benchmark endpoints

**What was built**
- `HNSW.save()`/`HNSW.load()` (`core/hnsw.py`) — pickle-based, since every
  attribute (`self.vectors`, `self.graph`, `self._rng`, plain ints/floats)
  is already ordinary, safely-picklable data.
- `FaissHNSW.save()`/`FaissHNSW.load()` (`core/faiss_index.py`) —
  `faiss.write_index`/`read_index` for the index itself, plus a small
  pickled sidecar file for `self._ids` (FAISS has no concept of external
  ids). `load()` bypasses `__init__` via `cls.__new__(cls)` since
  `__init__` always creates a brand-new empty index.
- `scripts/build_and_save_indices.py` — builds both indices ONCE on the
  full real 100k corpus and persists them to `data/` (gitignored,
  regenerable).
- `backend/app/main.py`: `create_app(hnsw_index=None, faiss_index=None,
  id_to_text=None, benchmark_results=None)` factory — loads the real
  persisted artifacts when an argument is omitted, or uses an injected
  fixture when provided (for tests).
- `backend/app/routers/`: `query.py` (`POST /api/query` — the real RAG
  pipeline), `compare.py` (`POST /api/compare` — same live query against
  both our HNSW and FAISS, side by side, no LLM call), `benchmark.py`
  (`GET /api/benchmark` — serves the precomputed `comparison_full100k.json`
  as is).
- `tests/test_api.py` — `fastapi.testclient.TestClient` against a
  `create_app()` built with small, fast, real (3,000-vector corpus subset)
  injected indices.
- `requirements.txt` gained `fastapi`, `uvicorn`.

**Decisions made, and why**
- Index persistence was unavoidable, not optional: Batch 6 measured a full
  100k build at ~216-225s (HNSW) and ~121-124s (FAISS). A server that
  rebuilt on every restart would be unusable. This was already flagged as
  a Batch 7 deferred item; Batch 8 is where it became load-bearing.
- `create_app()` is a factory, not a bare module-level `app = FastAPI()`
  (BuildBoard's own pattern). The factory takes optional pre-built
  indices/results and only falls back to loading the real ~174MB/168MB
  persisted artifacts from disk when an argument is `None`. This is what
  let `test_api.py` run against small, fast, *real* indices instead of
  needing the full production build (or a slow, disk-dependent test) just
  to exercise the routing/response logic. Deliberately did NOT put
  `app = create_app()` at module level in `main.py` — that would trigger
  the real disk-loading fallback merely by *importing* the module, which
  is exactly what `test_api.py` needs to avoid. The real server is run via
  uvicorn's factory mode (`uvicorn app.main:create_app --factory`) instead.
- `POST /api/compare` returns both indices' results with NO Claude call —
  kept separate from `/api/query` (which does call Claude/mock) because
  comparison is about retrieval, and CLAUDE_CONTEXT.md's own "Why We Are
  Building This" explicitly describes the demo goal as "query box ->
  retrieved chunks -> generated answer -> side-by-side comparison against
  FAISS on the same query" — two distinct concerns, two endpoints.
- `GET /api/benchmark` serves the already-measured `comparison_full100k.json`
  as a static read, not a live re-run. Rerunning the full ef-sweep
  benchmark per request would take several minutes and make the reported
  numbers non-reproducible/inconsistent across requests under any real
  load — the resume-payload numbers come from one controlled, logged run
  (Batch 6), not from whatever a given HTTP request happens to measure.
- `/api` prefix on resource routes mirrors BuildBoard's own reasoning
  exactly: keeps API paths distinct from whatever client-side routes the
  Batch 9 frontend adds, once both are served from the same domain.

**Bug found — NumPy scalar ids not JSON-serializable through FastAPI**
- **What broke**: `POST /api/query` crashed with `ValueError:
  [TypeError("'numpy.int64' object is not iterable"), TypeError('vars()
  argument must have __dict__ attribute')]` inside FastAPI's
  `jsonable_encoder`.
- **How it was discovered**: not by inspection — the very first real
  `test_api.py` run against real corpus ids (loaded from
  `corpus_metadata.jsonl` via `np.asarray(ids)`, same as every other
  script in this project) hit it immediately.
- **Root cause**: `HNSW.search()` and `FaissHNSW.search()` both returned
  `node_id` exactly as it was stored internally — a `numpy.int64` when the
  index was built from a NumPy ids array (true for the real corpus, and
  for every test in this project that mirrors it). FastAPI's
  `jsonable_encoder` doesn't know how to serialize a raw NumPy scalar.
  This exact class of bug was already anticipated and fixed once before —
  Batch 3's `BruteForce.search()` explicitly converts ids via `.item()`
  for precisely this reason, and its own log entry says so: "matters for
  JSON serialization once this is exposed via the API in Batch 8." That
  fix was never propagated to `HNSW.search()`/`FaissHNSW.search()` because
  nothing needed to JSON-serialize their output until now.
- **Fix**: both `HNSW.search()` and `FaissHNSW.search()` now convert
  `node_id` back to its native Python type (`node_id.item() if
  hasattr(node_id, "item") else node_id`) before returning, matching
  `BruteForce.search()`'s existing convention exactly. Distances were
  already native Python floats in both (`float(...)` was already applied),
  so only the id side needed the fix.
- **Verified the fix**: full test suite (all 8 test files) re-run
  afterward, all pass; `test_api.py`'s real requests (built from real
  NumPy corpus ids) now serialize correctly.

**How it was tested against this batch's actual goal**
- `tests/test_api.py` (4 checks, real 3,000-vector corpus subset, both
  indices real, injected via `create_app()`):
  - `GET /health` returns 200.
  - `POST /api/query`: real retrieval + labeled mock answer (no
    `ANTHROPIC_API_KEY` in this environment, same gap as Batch 7).
  - `POST /api/compare`: both indices returned real text; asserted >=3/5
    overlap as a floor (actual overlap in the real run: 5/5).
  - `GET /api/benchmark`: serves exactly the injected results object, no
    live computation.
- **Real production build**: `python scripts/build_and_save_indices.py`
  actually run on the full 100k corpus — HNSW build 225.1s, FAISS build
  121.3s (both consistent with Batch 6's measurements), saved to
  `data/hnsw_index.pkl` (174.2MB), `data/faiss_index.bin` (168.0MB),
  `data/faiss_ids.pkl` (1.9MB).
- **Real end-to-end server verification against the real persisted
  artifacts** (not the small test fixtures): `create_app()` with no
  arguments loaded the actual 100k-corpus indices from disk in **3.3
  seconds** (versus the ~3.5+2 minute build this replaces).
  `POST /api/compare` with `{"query": "the history of the internet"}`
  against the real full corpus returned the exact same top result from
  both indices — "ARPANET, the early ancestor of the internet, was
  designed to survive a nuclear war..." — genuinely correct and relevant,
  not a synthetic/subset check.

**Deferred to later batches**
- Verifying the real (non-mock) Claude API call — still blocked on not
  having an `ANTHROPIC_API_KEY` in this environment; same open item as
  Batch 7.
- Frontend (Batch 9) to actually call these three endpoints from a UI.
- Rate limiting / auth on the API — not needed for a local/portfolio demo;
  revisit only if this is actually deployed publicly (Batch 10).
```

### Batch 9: Frontend demo UI

**What was built**
- `frontend/`: React 19 + TypeScript + Vite, matching BuildBoard's stack
  exactly (Tailwind v4 via `@tailwindcss/vite`, axios, recharts, Inter/
  JetBrains Mono via `@fontsource`) minus `react-router-dom` — a single
  page needs no client-side routing.
- `src/api.ts` / `src/types.ts`: typed axios client for the three Batch 8
  endpoints, `VITE_API_URL` env var (defaults to `localhost:8000`), same
  convention as BuildBoard's frontend build.
- `src/components/QueryForm.tsx`, `AnswerPanel.tsx`, `CompareView.tsx`,
  `BenchmarkSection.tsx`, orchestrated from `src/App.tsx`: a query box; the
  real RAG answer + retrieved passages; a live our-HNSW-vs-FAISS
  side-by-side compare view with agreement markers; and the real Batch 6
  benchmark numbers as a `recharts` recall-vs-ef chart plus a table.

**Decisions made, and why**
- No automated frontend test suite added. Per general engineering practice
  for UI work (and this being a portfolio demo, not a product with users
  to protect from regressions), the right verification here is actually
  driving the app in a browser and looking at what renders — which is
  what was done (see below) — not writing component tests around a UI
  that's still likely to be visually iterated on.
- `/api/query` and `/api/compare` are called together via `Promise.all`
  when the user searches, not sequentially — they're independent
  (retrieval-only comparison doesn't need the Claude answer) and both are
  needed to render the page, so there's no reason to wait on one before
  starting the other. This choice is exactly what surfaced the batch's one
  real bug (below) — a case where doing the natural, correct frontend
  thing exposed a backend concurrency bug that a sequential-only usage
  pattern would never have hit.

**Bug found — concurrent embedding calls crashed the entire server process**
- **What broke**: driving the real UI in a headless browser (see below) —
  typing a query and clicking Search — hung indefinitely with no response.
  Investigating with direct concurrent `curl` requests (bypassing the
  frontend to isolate the cause) showed the *first* concurrent hit produced
  a hang with the server process pegged at 200%+ CPU; a later run's server
  process **disappeared entirely with no Python traceback** in its log,
  right after printing `Batches: 0%`. That combination — no traceback, mid
  native-library-call — is the signature of a native-level crash (e.g. a
  segfault inside PyTorch/Accelerate's BLAS backend), not a Python-level
  exception or deadlock.
- **How it was discovered**: not by inspection — the general UI-verification
  practice of actually launching the app and driving it in a browser (per
  this session's `run` skill / Playwright, since `chromium-cli` wasn't
  available in this environment) is what surfaced it. A curl-only,
  sequential test of each endpoint (done first, during Batch 8) never
  would have caught this, since the bug only manifests when
  `/api/query` and `/api/compare` are hit at the same moment — exactly
  what the real frontend does via `Promise.all`, and exactly what Batch
  8's own tests never happened to do.
- **Root cause**: `core/embed.py`'s `get_model()` lazily constructs a
  module-global `SentenceTransformer` on first use, with no synchronization.
  FastAPI runs sync route handlers (`query.py`/`compare.py`, both plain
  `def`, not `async def`) in a thread pool, so a single frontend action
  that fires both endpoints at once runs their `embed_texts()` calls on
  two separate threads simultaneously. On a cold model, both threads saw
  `_model is None` and both called `SentenceTransformer(MODEL_NAME)`
  concurrently. A first attempted fix (a lock around construction only,
  double-checked locking) reduced but did not eliminate the problem —
  concurrent *inference* (`model.encode()`) calls after construction could
  still crash the process, since sentence-transformers' CPU inference path
  isn't documented as safe to call from multiple threads simultaneously
  either.
- **Fix**: widened the lock in `core/embed.py` to cover the entire body of
  `embed_texts()` — both lazy construction and every `encode()` call are
  now fully serialized process-wide via one `threading.Lock()`. This
  trades a small amount of parallelism (two concurrent embedding requests
  now queue instead of running side by side) for a server that cannot
  crash from this, which is the right tradeoff for a single-machine
  portfolio demo backed by a CPU-only embedding model that takes low
  milliseconds per short query anyway.
- **Verified the fix**: ran the exact concurrent request pattern the
  frontend uses (`curl` firing `/api/query` and `/api/compare`
  simultaneously) across 4 separate trials after the fix — all 8 requests
  returned `200`, and `/health` remained responsive after every trial.
  Full backend test suite (all 8 test files) re-run afterward, all pass.
  Re-ran the real headless-browser flow end to end afterward with zero
  console errors.

**How it was tested against this batch's actual goal**
- `npm run build` (TypeScript typecheck + Vite production build) — passes
  clean.
- **Real browser verification** (the standard for UI work — actually
  running the app, not just type-checking it): started the real FastAPI
  backend (loaded from the actual persisted 100k-corpus indices, not test
  fixtures) and the real Vite dev server, then drove a headless Chromium
  (via Playwright, since `chromium-cli` wasn't available in this
  environment) through the golden path: load the page, type "What is the
  greenhouse effect?", click Search, and screenshot the result.
  - The benchmark section rendered correctly on page load — real numbers
    (216.4s/123.5s build time, 390.0MB/159.4MB memory, the real 5-row
    recall/latency table, and a working `recharts` line chart) fetched
    live from `GET /api/benchmark`.
  - After searching: the answer panel showed 5 real, genuinely relevant
    retrieved passages about the greenhouse effect (fossil fuels,
    greenhouse gases, volcanic CO2 history) and the labeled `[MOCK]`
    answer citing their real ids.
  - The compare view showed **5/5 agreement** between our HNSW and FAISS
    on the same live query, with matching passage text displayed
    side by side and agreement checkmarks rendered correctly.
  - Zero console errors (`page.on('console')`/`pageerror` listeners empty)
    after the full flow completed.

**Deferred to later batches**
- Verifying the real (non-mock) Claude answer renders correctly in the UI
  — still blocked on no `ANTHROPIC_API_KEY` in this environment; same open
  item carried from Batches 7-8. The UI code path for a real answer is
  identical to the mock path (`result.answer` is just a string either way)
  so no UI change is expected to be needed, but this is unverified.
- Batch 10 (deployment) — optional per the roadmap.
```

### Post-Batch-9: Exhaustive review pass (requested explicitly, not a numbered batch)

At the user's request, ran a multi-angle automated review (8 parallel review
agents: line-by-line, removed/missing behavior, cross-file tracing, reuse,
simplification, efficiency, altitude/root-cause, and CLAUDE.md-convention
checks) across the entire codebase, then verified every credible finding by
direct reproduction before fixing anything -- consistent with this
project's standing rule of never accepting a claimed bug (or a claimed
fix) without a real, run command proving it.

**Confirmed and fixed (each verified broken before, and fixed after):**

1. **`FaissHNSW.search()` had an unguarded race on shared mutable state.**
   `self.index.hnsw.efSearch = ef` was set as a side effect on the single
   `app.state.faiss_index` instance shared across all requests, with no
   lock -- two concurrent `/api/compare` requests with different `ef`
   values could interleave, one silently searching with the other's `ef`.
   Verified the underlying race directly with a minimal, unwrapped
   reproduction (raw `faiss.IndexHNSWFlat`, no wrapper) at ~1/800 with 4
   racing threads per side. Fixed with a `threading.Lock()`
   (`self._search_lock`) wrapping the entire set-then-search critical
   section in `core/faiss_index.py` -- the same bug class Batch 9's
   `core/embed.py` fix already addressed for the embedding model. A
   regression test (`check_concurrent_search_different_ef_no_corruption`,
   `tests/test_faiss_index.py`) hammers this with 4 threads x 200
   iterations per side; it's an honest stress test, not a guarantee of
   catching the fix's removal (the race is narrow enough that the same
   trial count didn't reliably reproduce it through the wrapper's exact
   call path) -- the real correctness guarantee is the lock covering the
   full critical section, verified by direct code inspection.
2. **`k<=0` crashed FAISS and silently misbehaved in HNSW.** Verified
   directly: `faiss.IndexHNSWFlat.search(q, k=0)` raises a raw, uncaught
   C++ `AssertionError` (would surface as an unhandled 500 through
   `/api/compare`). Separately, `HNSW.search()`'s `candidates[:k]` with a
   negative `k` silently sliced from the *end* (Python slice semantics) --
   verified `k=-1` against a 20-node index returned 19 (nearly all)
   results instead of erroring or returning none. Fixed both:
   `FaissHNSW.search()` now returns `[]` for `k<=0` before touching FAISS
   (matching `BruteForce.search()`'s existing Batch 3 convention);
   `HNSW.search()` does the same, and floors `ef` at 1 for the same
   reason. Regression tests added to both `tests/test_faiss_index.py` and
   `tests/test_search.py`.
3. **No input validation at the actual network boundary.** `QueryRequest`/
   `CompareRequest` accepted any integer for `k`/`ef` with no bounds --
   meaning a client could also send an arbitrarily large `ef` (e.g.
   `ef=5,000,000`), and since these are sync `def` routes running in
   FastAPI's bounded thread pool, enough such requests could occupy every
   worker thread and stall the server for everyone else. Consolidated the
   two identical request models into one `RetrievalRequest`
   (`backend/app/schemas.py`, also fixing the duplication finding below)
   with `k: Field(gt=0, le=100)` / `ef: Field(gt=0, le=2000)`. Verified
   against the real server: `k=0`, `k=-1`, `ef=0`, `ef=-10`, and
   over-bound values on both `/api/query` and `/api/compare` now return a
   clean `422`, not a `500` or silently wrong data.
4. **Unhandled `KeyError` risk if a persisted index and `corpus_metadata.jsonl`
   ever fall out of sync.** `core/rag.py`'s `retrieve()` does a raw
   `id_to_text[node_id]` lookup with no fallback. A silent `.get()`
   fallback was considered and rejected -- it would hide a real
   data-integrity bug instead of surfacing it, exactly backwards for a
   project about honest measurement. Instead, added
   `_check_ids_covered()` to `backend/app/main.py`'s `create_app()`,
   which fails loudly at *startup* (not per-request) if either index's
   ids aren't a subset of `id_to_text`'s keys -- mirroring
   `scripts/build_and_save_indices.py`'s own `len(ids)==len(embeddings)`
   check for the same class of drift, just applied to the two artifacts
   that are actually loaded together at serve time. Verified with a test
   that deliberately breaks this invariant and confirms `create_app()`
   raises `RuntimeError` before the server ever starts.
5. **`BruteForce` (the ground-truth oracle every recall number in this
   project is measured against) had no length check between `ids` and
   `vectors`.** A caller-side off-by-one would previously raise a
   confusing `IndexError` far from the real cause, or silently return a
   wrong id. Added an explicit `ValueError` at construction.
6. **`HNSW.insert()` never validated vector dimension against `self.dim`.**
   `self.dim` was stored but never read again anywhere in the file. A
   wrong-dimension vector was accepted silently, failing later (if at
   all) as a confusing numpy broadcast error deep inside `_distance`.
   `FaissHNSW` already gets this check for free from FAISS's C API; now
   both index types fail the same way, immediately, at `insert()`.
7. **Bare `assert` used for data-integrity checks that `python -O` (or
   `PYTHONOPTIMIZE=1`, plausibly set by a CI/deploy pipeline) compiles
   out entirely.** `scripts/build_and_save_indices.py` and
   `scripts/run_benchmark.py` each independently duplicated the same
   `load_corpus()` logic, gated behind a bare `assert`. Consolidated into
   `core/corpus.py`'s new `load_saved_corpus()` (also fixing the
   duplication finding below), which raises real `FileNotFoundError`/
   `ValueError` exceptions with actionable messages instead.
8. **Frontend `mergeSweeps` paired the two indices' ef-sweep data by array
   index, not by the `ef` value each point actually carries.**
   `scripts/run_benchmark.py` itself only trusts this pairing with an
   explicit `assert ours["ef"] == theirs["ef"]` before zipping (visible in
   its own printed comparison table) -- the frontend had no equivalent
   safety, so a benchmark JSON with mismatched-length or reordered sweeps
   would have silently paired the wrong `ef` values with no error. Fixed
   `BenchmarkSection.tsx`'s `mergeSweeps` to key-match by `ef` via a `Map`,
   throwing explicitly if a sweep point is missing on one side.
9. **`/api/compare`'s FAISS branch hand-rolled `core/rag.py`'s `retrieve()`
   dict-building logic instead of reusing it**, meaning a future fix to
   `retrieve()` could silently apply to only the HNSW side. Both branches
   now call the same `retrieve()`.
10. **`/api/query` left `embed_texts`'s default `show_progress=True` on**,
    spinning up a tqdm bar for a single-string embed while holding
    `core/embed.py`'s process-wide `_model_lock` -- extending how long
    every other concurrent request has to wait for it. `compare.py`
    already knew to pass `show_progress=False`; `answer_query()` now does
    too.
11. **A fresh `Anthropic()` HTTP client (and connection pool) was
    constructed on every non-mock `/api/query` request** instead of once
    per process. Now a lazily-constructed, reused module-level singleton
    in `core/rag.py`.

**Verified NOT a bug, despite looking similar:** `BruteForce.search()`'s
unconditional `self.ids[i].item()` (vs. the defensive `hasattr(x, "item")`
guards in `HNSW`/`FaissHNSW`) looked inconsistent on first read, but isn't
a live bug -- `BruteForce.__init__` always forces `self.ids =
np.asarray(ids)`, so every element is guaranteed to be a numpy scalar with
an `.item()` method; `HNSW`/`FaissHNSW` make no such guarantee about
caller-supplied ids, so their defensive guards are actually necessary and
`BruteForce`'s isn't. Also caught and corrected a bug in this review's own
first attempt at a regression test (`check_concurrent_search_different_ef_no_corruption`):
it initially compared FAISS's returned string ids (`"doc_0"`, ...) against
raw integer array positions from `np.argsort`, which can never be equal
regardless of any race -- the resulting "100% failure" was the test being
wrong, not the (correctly-implemented) lock; the lock's correctness was
re-verified with a fixed comparison and, separately, by removing the lock
and confirming a real (raw, unwrapped) reproduction of the race.

**Noted but deliberately not fixed this pass (architectural, not
correctness bugs -- flagged for a future decision, not unilaterally
applied):**
- The `sys.path.insert(0, ...)` project-root bootstrap is copy-pasted at
  varying `.parent` depths across `backend/app/main.py`, both routers, and
  all three `scripts/`. Fragile (a directory move silently breaks the
  parent-count in whichever file wasn't updated) but working today. A
  proper fix (a `pyproject.toml` + editable install, or one shared
  root-finder) is a real improvement but a structural one, not a bug fix.
- CORS is hardcoded to `http://localhost:5173` in `backend/app/main.py`,
  even though `frontend/src/api.ts` already supports a configurable
  `VITE_API_URL` for a real deployment. Not fixed now since deployment
  (Batch 10) hasn't happened and isn't required -- revisit if/when it does.
- HNSW and FAISS builds run sequentially in `scripts/run_benchmark.py`/
  `build_and_save_indices.py` though nothing after `ground_truth` is
  computed makes them dependent -- running them in parallel processes
  would cut total build time roughly in half. Not done: would add
  multiprocessing complexity to scripts whose current ~5-6 minute runtime
  is a one-time cost, not a per-request cost.
- `core/corpus.py`'s raw-download cache validity check
  (`_download_prefix`) only checks line count, not whether a prior
  download completed cleanly -- a narrow edge case in a one-time data
  pipeline script, not the live-serving path.
- Every user search fires `/api/query` and `/api/compare` concurrently
  from the frontend, each independently embedding the identical query
  string -- and since Batch 9's `_model_lock` now serializes all embedding
  calls process-wide, the two calls queue rather than run in parallel,
  effectively paying for two embedding passes back-to-back. A combined
  endpoint (or a request-scoped embedding cache) would remove the
  duplicate work, but changes the API's shape -- a design decision for the
  user to weigh in on, not applied unilaterally here.

**How all of this was verified:** every "confirmed" finding above was
independently reproduced against real, running code (not just read and
trusted) before any fix was written, and every fix was re-verified the
same way afterward -- including at full production scale (the real
100k-vector persisted indices, not just small test fixtures): a fresh
server load, then live `curl` requests reproducing each of the k=0,
negative-k, and over-bound-ef cases against the running server, confirming
clean `422`s where there were previously `500`s or silently wrong data,
and a final concurrent-load stress check confirming the server survives.
Full backend suite (all 8 test files, several new regression checks added)
passes; frontend TypeScript build passes.
```

### Live Claude API verification (closes the last open item from Batches 7-9)

A real `ANTHROPIC_API_KEY` was added to `.env` and `AI_MOCK` set to `false`.
This immediately surfaced three tests (`tests/test_rag.py`'s
`check_mock_answer_is_labeled_and_uses_real_ids` and
`check_retrieve_and_answer_query_end_to_end_real_corpus`, and
`tests/test_api.py`'s `check_query_endpoint_real_retrieval_mock_answer`)
that had assumed `AI_MOCK` would always default to `True` in this
environment -- true when they were written (no `.env` existed yet), false
now. Fixed by having each of those specific checks force `settings.ai_mock
= True` for their own duration (save/restore), since their actual purpose
is verifying retrieval correctness and the mock contract deterministically
and for free, not exercising the live API on every test run -- a separate
check (`check_live_claude_call_if_key_available`) already existed
specifically to test the live path when a real key is present.

**Real live Claude call, verified working**, both via the test and
directly against the real running server with the full 100k-corpus index:
query "What is the greenhouse effect?" returned a genuinely well-grounded,
multi-paragraph answer that cited specific retrieved passages by number
and correctly distinguished the natural greenhouse effect from its
human-caused amplification -- exactly the grounded-answer behavior the
system prompt (`core/rag.py`'s `_SYSTEM_PROMPT`) was designed to produce.
Separately, on the smaller 3,000-vector test-subset corpus (which happened
not to contain a passage directly defining the greenhouse effect), Claude
correctly said the provided passages didn't contain enough information to
fully answer the question, rather than filling the gap with outside
knowledge -- a real, unprompted demonstration that the "answer ONLY from
context, say so if you can't" instruction actually works as intended.

Full backend suite (all 8 test files) re-run afterward with the real key
active, all pass.
