# Local Showcase Dashboard

Static visualization of the completed **Graph-Augmented Code Intelligence Engine** and its canonical evaluation results.

## Purpose

This dashboard is a **presentation layer** for the M0–M10 portfolio project. It does not run Tree-sitter, FAISS, MiniLM, SQLite, or Python in the browser. It fetches **committed canonical JSON** produced by the real `aicode` CLI and renders metrics, architecture, and evaluation summaries for recruiter review (~30–90 seconds).

## Run locally

From the **repository root** (not from `demo/`):

```bash
cd /path/to/graph_code_intelligence
python3 -m http.server 8080
```

Open:

**http://localhost:8080/demo/**

### Why serve from repo root?

The dashboard loads canonical artifacts at runtime:

- `../benchmarks/results/python_retrieval_v1_minilm.json`
- `../benchmarks/results/incremental_work_v1.json`

These paths resolve correctly when the static server root is the repository, not the `demo/` folder.

If port 8080 is occupied, use another port (e.g. `8081`) and open `http://localhost:8081/demo/`.

## What this is / is not

| Yes | No |
|-----|-----|
| Static HTML/CSS/JS | Backend API |
| Canonical committed metrics | Fake or invented benchmark numbers |
| Architecture & evaluation storytelling | Engine execution in browser |
| Python built-in HTTP server | Node, npm, build step, framework |

## Files

| File | Role |
|------|------|
| `index.html` | Semantic page structure |
| `styles.css` | Dark engineering showcase theme |
| `app.js` | Fetch & render canonical JSON |
| `README.md` | This file |

## Data source of truth

All benchmark and incremental work metrics come from:

- `benchmarks/results/python_retrieval_v1_minilm.json`
- `benchmarks/results/incremental_work_v1.json`

If fetch fails (e.g. opened as `file://` or wrong server root), the page shows an error card with run instructions — it does **not** substitute fake metrics.

## Engine unchanged

Creating or editing files under `demo/` does not modify:

- `src/codeintel/` retrieval semantics
- benchmark corpus or `benchmark.json`
- canonical result artifacts
- ranking constants

Reranking explainability and context-compiler examples shown on the page include **static content verified by the final local demo** where those details are not stored in benchmark JSON.
