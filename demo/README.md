# Local Showcase Dashboard

Static visualization of the completed **Graph-Augmented Code Intelligence Engine** and its canonical evaluation results.

## Purpose

This dashboard is a **presentation layer** for the M0–M10 portfolio project. It does not run Tree-sitter, FAISS, MiniLM, SQLite, or Python in the browser. It loads **canonical JSON** (copied from committed evaluation artifacts) and renders metrics, architecture, and evaluation summaries for recruiter review (~30–90 seconds).

## Run locally

From the `demo/` directory:

```bash
cd demo
python3 -m http.server 8080
```

Open:

**http://localhost:8080/**

If port 8080 is occupied, use another port (e.g. `8081`).

## Deploy on Vercel

| Setting | Value |
|---------|--------|
| Framework Preset | **Other** |
| Root Directory | **demo** |
| Build Command | *(none)* |
| Output Directory | *(none — static files at root)* |

- No backend
- No build step
- No Node dependency

The showcase is fully self-contained under `demo/` — Vercel does not need to detect or deploy the Python engine.

## What this is / is not

| Yes | No |
|-----|-----|
| Static HTML/CSS/JS | Backend API |
| Canonical committed metrics | Fake or invented benchmark numbers |
| Architecture & evaluation storytelling | Engine execution in browser |
| Python built-in HTTP server (local) | Node, npm, build step, framework |

## Files

| File | Role |
|------|------|
| `index.html` | Semantic page structure |
| `styles.css` | Dark engineering showcase theme |
| `app.js` | Fetch & render canonical JSON |
| `data/*.json` | Self-contained copies of committed evaluation artifacts |
| `README.md` | This file |

## Data source of truth

Runtime metrics load from:

- `data/python_retrieval_v1_minilm.json`
- `data/incremental_work_v1.json`

These are exact copies of the committed canonical artifacts at:

- `benchmarks/results/python_retrieval_v1_minilm.json`
- `benchmarks/results/incremental_work_v1.json`

If fetch fails (e.g. opened as `file://`), the page shows an error card — it does **not** substitute fake metrics.

## Engine unchanged

Files under `demo/` do not modify:

- `src/codeintel/` retrieval semantics
- benchmark corpus or `benchmark.json`
- canonical result artifacts in `benchmarks/results/`
- ranking constants

Reranking explainability and context-compiler examples shown on the page include **static content verified by the final local demo** where those details are not stored in benchmark JSON.
