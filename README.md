# Graph-Augmented Code Intelligence Engine

A graph-augmented code intelligence engine for structural code retrieval, ranking, and context compilation over large codebases.

## Overview

This project is building a code intelligence engine that combines:

- static program structure
- information retrieval
- graph algorithms
- structured ranking
- context engineering

The LLM is not the engine itself. The engine is designed to work and be benchmarkable without requiring an LLM. An LLM, when used, consumes structured context produced by the engine rather than driving retrieval directly.

## Why This Project Exists

Traditional code RAG systems often treat source code primarily as text chunks. That approach ignores the structural relationships that engineers use when navigating a codebase: definitions, references, imports, call patterns, and module boundaries.

This project investigates whether structural program information can improve retrieval and reasoning over large codebases.

**Central question:** Can structural program information improve code retrieval compared with text-only retrieval?

That question has not been answered yet. This repository documents the engineering path toward a measurable evaluation.

## Core Technical Thesis

The planned evaluation compares retrieval strategies along an ablation ladder:

1. **Lexical retrieval** — keyword and phrase matching over source text
2. **Dense retrieval** — embedding similarity over code or text representations
3. **Hybrid retrieval** — fusion of lexical and dense signals
4. **Hybrid + Code Graph** — graph expansion over static relationships between symbols and files
5. **Hybrid + Graph + Structured Reranking + Context Compilation** — full engine with token-budget-aware context assembly

This ladder is the planned ablation and evaluation direction. Milestone 5 adds **graph-augmented hybrid retrieval** as a fourth comparable method. Lexical, dense, and hybrid baselines remain unchanged and graph-independent. Structured reranking and explainability remain Milestone 6.

## Target Architecture

The following diagram describes the **target** end-to-end pipeline. It is not current functionality.

```mermaid
flowchart TD
    A[Repository] --> B[Language Adapter]
    B --> C[Static Analysis]
    C --> D[Files / Symbols / Relationships]
    D --> E[Code Graph]
    E --> F[Lexical Index + Dense Index]
    F --> G[Candidate Fusion]
    G --> H[Graph Expansion]
    H --> I[Structured Reranker]
    I --> J[Context Compiler]
    J --> K[Structured Context]
    K --> L[CLI / Optional LLM]
```

## Current Status

The project is at **Milestone 5 — Graph-Augmented Retrieval**.

Verified capabilities today:

- Python 3.14 project managed with [uv](https://docs.astral.sh/uv/)
- `src/`-layout Python package (`codeintel`)
- Typer CLI with `aicode version`, `aicode inspect`, `aicode graph`, `aicode index`, `aicode embed`, and `aicode search`
- Tree-sitter Python parsing isolated behind a thin parser wrapper
- `PythonAdapter` semantic extraction into language-neutral `Symbol` and `CodeUnit` models
- Repository-level symbol indexing and duplicate qualified-name detection
- Static `CONTAINS`, `IMPORTS`, `REFERENCES`, `CALLS`, and `INHERITS` relations
- Conservative Python lexical-scope analysis with `RESOLVED` / `PROBABLE` / `UNRESOLVED` status
- In-memory `CodeGraph` with incoming/outgoing adjacency, filtering, bounded BFS, and shortest distance
- SQLite persistent index (`PRAGMA user_version = 1`) with transactional full rebuild
- Persisted files, Symbols, CodeUnits, and Relations using repository-relative POSIX paths
- FTS5 one-document-per-CodeUnit indexing with the `unicode61` tokenizer
- BM25 lexical retrieval with higher-is-better `SearchResult.score` (`score = -raw_bm25`)
- Safe ordinary-text query construction (quoted tokens joined with `OR`)
- Optional `SymbolKind` and path-prefix search filters
- `EmbeddingProvider` abstraction with optional Sentence Transformers provider
- Default local model `sentence-transformers/all-MiniLM-L6-v2` (general semantic baseline, not code-specialized)
- Deterministic offline fake embedding provider for tests (no model download in CI)
- FAISS `IndexFlatIP` exact dense search with L2-normalized cosine via inner product
- Dense artifacts under `.codeintel/dense/` (`index.faiss` + `metadata.json`)
- Vector identity by `symbol.qualified_name` ordinal mapping (not SQLite row ids)
- Whole-corpus fingerprint stale protection for dense artifacts
- Reciprocal Rank Fusion hybrid retrieval (`RRF_K = 60`, equal lexical/dense weights)
- Graph-augmented hybrid retrieval (`search --mode graph`) over persisted CodeGraph
- Graph seeds: top 10 Hybrid results; direct depth-1 RESOLVED neighbors only
- Included graph relations: `CALLS`, `REFERENCES`, `INHERITS`, `CONTAINS` (both directions)
- Excluded from graph retrieval: `IMPORTS`, `PROBABLE`, `UNRESOLVED`; MODULE nodes not returned
- Unique-seed structural support `Σ 1/(60 + seed_rank)` then final Hybrid+Graph RRF (`k=60`)
- `aicode search --mode lexical|dense|hybrid|graph` (default remains lexical)
- Deterministic module-name derivation and source-file discovery
- Fixture-backed parser, relation, graph, persistence, lexical, dense, hybrid, graph-retrieval, and CLI tests
- Unit tests with pytest
- Linting and formatting with Ruff
- Strict static typing with mypy
- GitHub Actions CI on push and pull request

Structured reranking, context compilation, incremental indexing, and LLM answer generation are not implemented yet.

## Current Capabilities

### CLI

Commands currently available:

```bash
uv run aicode --help
uv run aicode version
uv run aicode inspect PATH
uv run aicode graph PATH
uv run aicode graph PATH --symbol QUALIFIED_NAME
uv run aicode index PATH
uv run aicode embed PATH
uv run aicode search PATH QUERY
uv run aicode search PATH QUERY --mode dense
uv run aicode search PATH QUERY --mode hybrid
uv run aicode search PATH QUERY --mode graph
```

Expected version output:

```text
aicode 0.1.0
```

`aicode inspect` accepts a Python file or a directory. It discovers supported files, analyzes each through `PythonAdapter`, and prints module names, syntax-error status, Symbols, and CodeUnit summaries.

`aicode graph` expects a **repository directory**. It builds a repository-level symbol index and in-memory CodeGraph, then prints relation summaries. `--symbol` shows incoming and outgoing edges for one qualified name. This inspection command is separate from retrieval `--mode graph`.

`aicode index` expects a **repository directory**. It analyzes the repository, then writes a full SQLite snapshot to `PATH/.codeintel/index.db` by default (`--db` overrides). Indexing is a transactional full rebuild.

`aicode embed` builds/rebuilds the dense FAISS artifact from an **existing** SQLite index (it does not run `index` automatically). Default artifact directory: `PATH/.codeintel/dense/`. Requires the optional `embeddings` extra. Default model: `sentence-transformers/all-MiniLM-L6-v2`. Overrides: `--db`, `--dense-dir`, `--model`.

`aicode search PATH QUERY` searches a previously built index (default `PATH/.codeintel/index.db`, overridable with `--db`). It does not reindex automatically. Default `--mode` is `lexical` (unchanged from Milestone 3). Dense/hybrid/graph modes require a dense artifact (`aicode embed`) and the embeddings extra. Optional `--limit`, `--kind`, `--path-prefix`, and `--dense-dir` are supported.

### Language adapter and semantic extraction

`LanguageAdapter` defines language identity, supported extensions, file-support detection, and `analyze_file(...)`.

`PythonAdapter` currently extracts:

- module, class, function, and method Symbols
- nested functions (classified as `FUNCTION`, not `METHOD`)
- decorated and async definitions
- declaration signatures
- exact source spans and CodeUnit text (no MODULE CodeUnits)
- `has_syntax_errors` for malformed Python without crashing analysis

Tree-sitter `Tree` / `Node` objects do not leak into the semantic models or CLI output.

### Static relationships and CodeGraph

Repository analysis discovers Python files, runs Milestone 1 extraction, then builds a qualified-name symbol index **before** cross-file relation resolution.

Implemented relation kinds:

- **CONTAINS** — derived from existing Symbol parent links (`RESOLVED`)
- **IMPORTS** — module-level import edges; repo-local names resolve, stdlib/third-party stay `UNRESOLVED`
- **CALLS** — conservative lexical / import / `self` / `cls` call edges
- **REFERENCES** — only when a use maps to a concrete repository Symbol; unknown locals/builtins are omitted
- **INHERITS** — class base expressions, including multiple bases

`DEFINES` is **not** emitted. With a Symbol-only graph it would duplicate `CONTAINS` without distinct semantics. It is deferred until binding/file-level definition modeling exists.

Resolution is intentionally conservative:

- same-module and imported callees may be `RESOLVED`
- `self.method()` / `cls.method()` / `KnownClass.method()` are `PROBABLE` because of Python dynamic dispatch
- inherited methods are not followed through MRO; `self.base_method()` stays `UNRESOLVED` unless that method is declared on the current class
- star imports remain `UNRESOLVED` and do not bind names
- assignment aliases, `global` / `nonlocal`, runtime imports, `getattr`, and monkeypatching are not modeled
- there is no type inference and no execution of analyzed code
- each Python file is parsed twice by design (symbol extraction, then relations)

The in-memory CodeGraph uses `Symbol.qualified_name` as node identity. Unresolved relations are stored but do not create fake target nodes or participate in distance traversal.

### Persistent index and lexical retrieval

`IndexDatabase` stores a language-neutral snapshot of repository analysis in SQLite:

- schema version via `PRAGMA user_version` (currently `1`)
- tables for files, symbols, code_units, and relations
- repository-relative POSIX path identity for files
- one FTS5 document per CodeUnit (`unicode61` tokenizer)
- transactional full rebuild (delete snapshot, rewrite, commit; rollback preserves the prior index)

`search_code_units(...)` performs BM25 retrieval. SQLite's raw `bm25(...)` ranks lower/more-negative as better; `SearchResult.score` exposes `-raw_bm25` so **higher score means better relevance** within that method. Scores are retrieval ranks, not probabilities, and are **not comparable across** lexical / dense / hybrid modes.

Ordinary search queries are tokenized on whitespace, quoted as literal FTS phrases, and joined with `OR`. Raw FTS operator syntax is not exposed.

### Dense and hybrid retrieval

Dense retrieval embeds one document per CodeUnit using:

```text
symbol: <qualified_name>
signature: <signature or empty>
code:
<source_text>
```

`EmbeddingProvider` returns raw float32 vectors. The vector-index layer L2-normalizes documents and queries once, then searches with FAISS `IndexFlatIP` (exact cosine via inner product; higher is better). Semantic vector identity is `Symbol.qualified_name` stored in artifact metadata ordinal order — not SQLite `code_units.id`.

Dense artifacts live outside SQLite under `PATH/.codeintel/dense/`:

- `index.faiss`
- `metadata.json` (`artifact_version = 1`, provider/model/dimension, metric, fingerprint, `qualified_names`, …)

A whole-corpus SHA-256 fingerprint protects against silently searching a stale dense artifact after the SQLite snapshot changes. Rebuild with `aicode embed PATH`.

Hybrid retrieval fuses lexical and dense ranked lists with Reciprocal Rank Fusion:

- `RRF_K = 60`
- ranks start at 1
- equal lexical and dense weights
- per-retriever candidate depth `max(50, 5 * limit)`
- deduplicate by `symbol_qualified_name`
- graph relations are **not** used inside Hybrid (ablation integrity)

### Graph-augmented retrieval

`--mode graph` means **graph-augmented Hybrid**, not a replacement of Hybrid:

1. Retrieve Hybrid base pool `max(10, limit)`
2. Take top `GRAPH_SEED_COUNT = 10` Hybrid seeds
3. Reconstruct `CodeGraph` from persisted SQLite Symbols/Relations (no live reparse)
4. Expand **direct** (depth-1) neighbors in **both** directions for `CALLS`, `REFERENCES`, `INHERITS`, `CONTAINS`
5. Keep only `RESOLVED` edges; exclude `IMPORTS`, `PROBABLE`, and `UNRESOLVED`
6. Return only Symbols with persisted CodeUnits; never return MODULE nodes
7. Aggregate unique supporting seed ranks: `GraphSupport = Σ 1/(60 + seed_rank)`
8. Rank structural candidates by GraphSupport, then fuse Hybrid ranking + structural ranking with equal-weight RRF (`k=60`)

Conservative policy rationale: keep ablation clean, bound expansion (no depth-2 / module fan-out), and avoid uncertain PROBABLE edges and noisy IMPORTS in the first structural baseline.

The real embedding stack is an **optional** extra:

```bash
uv sync --extra embeddings
```

Default `uv sync` (and CI) installs NumPy + FAISS only. Tests use a deterministic fake provider and do not download models.

Baseline limitations:

- indexing is full rebuild only (no incremental updates yet)
- dense artifacts are also full rebuild (no changed-file embedding invalidation)
- `unicode61` is not a code-specific tokenizer
- camelCase is not specially segmented
- default MiniLM model is a **general** semantic baseline, not code-specialized
- MiniLM may truncate long inputs according to its own max sequence length; one CodeUnit remains one vector (no chunking)
- FAISS baseline is exact `IndexFlatIP`, not ANN-tuned HNSW/IVF
- scores are not comparable across retrieval modes
- graph mode requires a dense artifact because Hybrid is the base
- graph expansion is direct one-hop only (no depth-2)
- only RESOLVED edges; IMPORTS and PROBABLE excluded from graph retrieval
- no relation-type-specific weights
- no structured reranking / user-facing graph explanations yet
- no cross-encoder yet
- no benchmark improvement claims yet

## Technology Stack

### Current

| Tool | Role |
|------|------|
| Python 3.14 | Runtime |
| uv | Dependency and environment management |
| Typer | CLI framework |
| Tree-sitter | Python syntax parsing |
| tree-sitter-python | Python grammar |
| SQLite / FTS5 | Persistent index and lexical BM25 retrieval (stdlib `sqlite3`) |
| NumPy | Dense vector arrays |
| FAISS (`faiss-cpu`) | Exact `IndexFlatIP` dense index |
| Sentence Transformers (optional extra `embeddings`) | Local embedding provider |
| pytest | Unit and integration testing |
| Ruff | Linting and formatting |
| mypy | Strict static type checking |
| Hatchling | Package build backend |
| Git | Version control |
| GitHub Actions | Continuous integration |

### Planned

| Technology | Planned role |
|------------|--------------|
| Structured reranker / explainability | Milestone 6 |
| Context compiler | Milestone 7 |
| Incremental indexing | Milestone 8 |
| C++ language adapter | Milestone 9 |
| Benchmark suite / optional LLM | Milestone 10 |

Sentence Transformers is optional and is **not** installed by default `uv sync`.

## Repository Structure

```text
graph_code_intelligence/
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   └── codeintel/
│       ├── __init__.py
│       ├── cli.py
│       ├── dense.py
│       ├── discovery.py
│       ├── embeddings.py
│       ├── graph.py
│       ├── graph_retrieval.py
│       ├── hybrid.py
│       ├── lexical.py
│       ├── models.py
│       ├── repository.py
│       ├── vector_index.py
│       ├── languages/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── python/
│       │       ├── __init__.py
│       │       ├── adapter.py
│       │       ├── parser.py
│       │       └── relations.py
│       └── storage/
│           ├── __init__.py
│           ├── database.py
│           └── schema.py
├── tests/
│   ├── fixtures/
│   │   ├── python_dense/
│   │   ├── python_graph/
│   │   ├── python_graph_search/
│   │   ├── python_repo/
│   │   └── python_search/
│   ├── helpers/
│   │   └── fake_embeddings.py
│   ├── integration/
│   │   ├── test_embed_search_cli.py
│   │   ├── test_graph_cli.py
│   │   ├── test_graph_search_cli.py
│   │   ├── test_index_search_cli.py
│   │   └── test_inspect_cli.py
│   └── unit/
│       ├── test_cli.py
│       ├── test_dense.py
│       ├── test_discovery.py
│       ├── test_embeddings_contract.py
│       ├── test_graph.py
│       ├── test_graph_retrieval.py
│       ├── test_hybrid.py
│       ├── test_language_adapter.py
│       ├── test_lexical.py
│       ├── test_models.py
│       ├── test_python_adapter.py
│       ├── test_python_parser.py
│       ├── test_python_relations.py
│       ├── test_repository.py
│       ├── test_storage.py
│       └── test_vector_index.py
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
├── pyproject.toml
└── uv.lock
```

uv creates a local `.venv/` directory during setup. That directory is gitignored and is not part of the tracked source tree.

## Development Setup

### Prerequisites

- Git
- [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.12.7

Python 3.14.7 is pinned in `.python-version`. uv installs and uses that version automatically.

From the repository directory:

```bash
uv sync
```

This command:

- resolves locked dependencies from `uv.lock`
- creates or updates the local `.venv/`
- installs the package and development dependencies (NumPy + FAISS included; Sentence Transformers is **not**)

To enable the real local embedding provider:

```bash
uv sync --extra embeddings
```

Do not manually `pip install` packages into the environment.

## CLI

```bash
uv run aicode --help
uv run aicode version
uv run aicode inspect PATH
uv run aicode graph PATH
uv run aicode graph PATH --symbol QUALIFIED_NAME
uv run aicode index PATH
uv run aicode embed PATH
uv run aicode search PATH QUERY
uv run aicode search PATH QUERY --mode dense
uv run aicode search PATH QUERY --mode hybrid
uv run aicode search PATH QUERY --mode graph
```

Expected version output:

```text
aicode 0.1.0
```

Example inspection of the parser fixture repository:

```bash
uv run aicode inspect tests/fixtures/python_repo
```

Example graph inspection:

```bash
uv run aicode graph tests/fixtures/python_graph
uv run aicode graph tests/fixtures/python_graph --symbol service.Service
```

Example lexical index and search against a temporary database:

```bash
uv run aicode index tests/fixtures/python_search --db /tmp/codeintel-search.db
uv run aicode search tests/fixtures/python_search "payment authorization" --db /tmp/codeintel-search.db
```

Example dense artifact build (requires `uv sync --extra embeddings`; may download the default MiniLM model once):

```bash
uv run aicode index tests/fixtures/python_dense --db /tmp/codeintel-dense.db
uv run aicode embed tests/fixtures/python_dense --db /tmp/codeintel-dense.db --dense-dir /tmp/codeintel-dense-art
uv run aicode search tests/fixtures/python_dense "check whether a login session is still valid" \
  --db /tmp/codeintel-dense.db --dense-dir /tmp/codeintel-dense-art --mode hybrid
```

Generated indexes default to `PATH/.codeintel/index.db` and dense artifacts to `PATH/.codeintel/dense/`. The `.codeintel/` directory is gitignored.

## Testing and Code Quality

Run each check independently:

```bash
uv run pytest
```

Runs the unit and integration test suites, including Python parsing fixtures, relationship extraction, CodeGraph APIs, SQLite persistence, lexical BM25 search, dense/hybrid/graph retrieval with a fake embedding provider, and CLI inspect/graph/index/embed/search behavior. Default pytest does **not** download embedding models.

```bash
uv run ruff check .
```

Runs Ruff lint rules configured in `pyproject.toml`.

```bash
uv run ruff format --check .
```

Verifies code formatting without modifying files. Use `uv run ruff format .` to apply formatting locally.

```bash
uv run mypy
```

Runs strict static type checking over `src/` and `tests/`.

## Continuous Integration

The workflow at `.github/workflows/ci.yml` validates on every push and pull request:

- dependency synchronization from the lock file
- pytest
- Ruff lint
- Ruff format check
- mypy

The workflow is configured in this repository and runs on GitHub for pushes and pull requests against the published project remote.

## Roadmap

| Milestone | Scope | Status |
|-----------|-------|--------|
| 0 | Repository Foundation | Complete |
| 1 | Python Parsing & Semantic Units | Complete |
| 2 | Static Relationships & Code Graph | Complete |
| 3 | Persistent Index & Lexical Retrieval | Complete |
| 4 | Dense & Hybrid Retrieval | Complete |
| 5 | Graph-Augmented Retrieval | **Current** |
| 6 | Structured Reranking & Explainability | Planned |
| 7 | Token-Budget Context Compiler | Planned |
| 8 | Incremental Indexing | Planned |
| 9 | C++ Language Adapter | Planned |
| 10 | Benchmark Suite, Optional LLM Integration & Portfolio Polish | Planned |

## Design Principles

- **Code structure over arbitrary chunking** — retrieval should respect program structure, not only text boundaries.
- **LLM as consumer, not core engine** — the engine produces structured context; an LLM is optional downstream.
- **Measurable improvements over AI buzzwords** — claims should be backed by benchmarks, not marketing language.
- **Interpretable retrieval and ranking** — ranking decisions should be inspectable and explainable where possible.
- **Incremental development** — each milestone adds a testable subsystem before the next layer is built.
- **Independently testable subsystems** — parsing, indexing, retrieval, and ranking can be validated in isolation.
- **No unnecessary distributed infrastructure** — a single-machine architecture is sufficient for the research scope.
- **Honest static-analysis limitations** — dynamic behavior, macros, and runtime indirection are acknowledged constraints.

## Evaluation Plan

Future evaluation will measure retrieval quality using standard information-retrieval metrics:

- **Recall@k** — fraction of relevant items retrieved in the top *k* results
- **MRR** (Mean Reciprocal Rank) — average reciprocal rank of the first relevant result
- **nDCG** (normalized Discounted Cumulative Gain) — ranking quality with position-weighted relevance

The benchmark will compare, at minimum:

- BM25 (lexical baseline)
- Dense retrieval
- Hybrid retrieval
- Hybrid + Graph
- Full Engine (hybrid + graph + reranking + context compilation)

**No benchmark results exist yet.** Numbers will be reported only after the benchmark suite is implemented and run.

## License

This project is released under the MIT License. See [MIT License](LICENSE) for the full text.
