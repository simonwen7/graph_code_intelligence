# Python Structural Retrieval Benchmark v1

Frozen labeled evaluation corpus for Milestone 10.

## Identity

- **benchmark_id:** `python-structural-retrieval-v1`
- **benchmark_version:** `1`
- **language:** Python only
- **queries:** 24 (6 lexical / 6 behavioral / 6 calls / 6 inheritance)
- **showcase cases (preselected before first real run):** `lex-01`, `calls-02`
- **SHA-256 of `benchmark.json`:**
  `5125c8facaa3344417ca5ea31958f2f6d1a1393a3675963c8e2cbf8d611ec2de`

## Corpus

`corpus/` is an 11-file synthetic order/payment/inventory application with
cross-file `CALLS`, `INHERITS`, `CONTAINS`, and `IMPORTS` under the existing
Python relation extractor. It intentionally mixes name-heavy and behavioral
retrieval opportunities so the benchmark is not graph-only.

## Label policy

- Gold labels are ordinary CodeUnit qualified names only (never MODULE).
- Relevance is binary (`relevant_qnames`); no graded labels.
- Queries and golds were authored and hashed before the first real MiniLM run.
- Do not edit corpus/queries/golds after that freeze to chase better numbers.
  Objective defects require a version bump.

## How to run

```bash
uv sync --extra embeddings
uv run aicode index benchmarks/python_retrieval_v1/corpus --language python --full
uv run aicode embed benchmarks/python_retrieval_v1/corpus
uv run aicode benchmark \
  benchmarks/python_retrieval_v1/corpus \
  benchmarks/python_retrieval_v1/benchmark.json \
  --json-out benchmarks/results/python_retrieval_v1_minilm.json \
  --markdown-out benchmarks/results/python_retrieval_v1_minilm.md
```

CI evaluates the harness with a fake embedding provider and does not download
MiniLM.

## Limitations

- Small synthetic corpus; not a universal code-search claim.
- Hand-authored labels may reflect author familiarity.
- MiniLM is general text, not code-specialized.
- This corpus has **no REFERENCES** edges under the Python extractor; structural
  evidence available here is CALLS / INHERITS / CONTAINS / IMPORTS.
- Aggregate Graph/Reranked metrics on the canonical MiniLM run underperformed
  Hybrid/Graph respectively; results are preserved as measured.
