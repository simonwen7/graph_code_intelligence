/**
 * Graph-Augmented Code Intelligence Engine — Showcase Dashboard
 * Loads committed canonical JSON; no engine execution in browser.
 */

const BENCHMARK_URL = "../benchmarks/results/python_retrieval_v1_minilm.json";
const INCREMENTAL_URL = "../benchmarks/results/incremental_work_v1.json";

const MODES = ["lexical", "dense", "hybrid", "graph", "reranked"];

const MODE_LABELS = {
  lexical: "LEXICAL",
  dense: "DENSE",
  hybrid: "HYBRID",
  graph: "GRAPH",
  reranked: "RERANKED",
};

const MODE_DESCRIPTIONS = {
  lexical: "FTS5 / BM25 text matching",
  dense: "MiniLM embeddings + FAISS FlatIP",
  hybrid: "Lexical + Dense via RRF",
  graph: "Hybrid + RESOLVED structural neighbors",
  reranked: "Graph base + relation-evidence RRF channels",
};

const METRIC_LABELS = {
  hit_at_1: "Hit@1",
  hit_at_5: "Hit@5",
  hit_at_10: "Hit@10",
  mrr_at_10: "MRR@10",
};

const SHOWCASE_QUERY_ID = "calls-02";

const SCENARIO_LABELS = {
  "no-op": "No-op",
  "body-edit": "Body edit",
  "symbol-rename": "Symbol rename",
  "add-delete": "Add + delete",
};

/** @type {object | null} */
let benchmarkData = null;

/** @type {string} */
let activeMetric = "mrr_at_10";

/**
 * @param {string} url
 * @returns {Promise<object>}
 */
async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} loading ${url}`);
  }
  return response.json();
}

/**
 * @param {string} elementId
 * @param {string} message
 */
function showError(elementId, message) {
  const el = document.getElementById(elementId);
  if (!el) return;
  el.classList.remove("hidden");
  el.innerHTML = `
    <strong>Canonical data could not be loaded.</strong>
    ${message}
    <code>python3 -m http.server 8080</code>
    <span style="display:block;margin-top:0.35rem;font-size:0.8125rem;">Then open http://localhost:8080/demo/</span>
  `;
}

/**
 * @param {number} value
 * @param {number} [digits=4]
 * @returns {string}
 */
function formatMetric(value, digits = 4) {
  return Number(value).toFixed(digits);
}

/**
 * @param {object} data
 */
function renderRetrievalShowcase(data) {
  const queryEl = document.getElementById("retrieval-query");
  const goldEl = document.getElementById("retrieval-gold");
  const cardsEl = document.getElementById("mode-cards");
  const flowEl = document.getElementById("rank-flow");

  if (!queryEl || !cardsEl || !flowEl) return;

  const showcase = data.queries.find((q) => q.id === SHOWCASE_QUERY_ID);
  if (!showcase) {
    queryEl.textContent = "Showcase query calls-02 not found in benchmark data.";
    return;
  }

  queryEl.textContent = `"${showcase.query}"`;
  if (goldEl && showcase.relevant_qnames?.[0]) {
    goldEl.textContent = showcase.relevant_qnames[0];
  }

  const ranks = showcase.ranks;
  const rankValues = MODES.map((m) => ranks[m]).filter((r) => r != null);
  const bestRank = Math.min(...rankValues);

  cardsEl.innerHTML = MODES.map((mode) => {
    const rank = ranks[mode];
    const isBest = rank === bestRank;
    return `
      <article class="mode-card${isBest ? " mode-card-best" : ""}">
        <div class="mode-card-label">${MODE_LABELS[mode]}</div>
        <div class="mode-card-rank">#${rank}</div>
        <p class="mode-card-desc">${MODE_DESCRIPTIONS[mode]}</p>
      </article>
    `;
  }).join("");

  const flowModes = ["dense", "hybrid", "graph", "reranked"];
  flowEl.innerHTML = flowModes
    .map((mode, i) => {
      const rank = ranks[mode];
      const best = rank === bestRank;
      const arrow = i < flowModes.length - 1 ? '<span class="rank-flow-arrow" aria-hidden="true">→</span>' : "";
      return `
        <span class="rank-flow-item">
          <span class="rank-flow-mode">${MODE_LABELS[mode]}</span>
          <span class="rank-flow-value${best ? " best" : ""}">#${rank}</span>
        </span>${arrow}
      `;
    })
    .join("");
}

/**
 * @param {object} data
 * @param {string} metric
 */
function renderBenchmarkChart(data, metric) {
  const chartEl = document.getElementById("benchmark-chart");
  if (!chartEl || !data.aggregate) return;

  const values = MODES.map((mode) => data.aggregate[mode][metric]);
  const max = Math.max(...values, 0.001);

  chartEl.setAttribute("aria-label", `${METRIC_LABELS[metric]} comparison by mode`);

  chartEl.innerHTML = MODES.map((mode) => {
    const value = data.aggregate[mode][metric];
    const pct = (value / max) * 100;
    return `
      <div class="chart-row">
        <span class="chart-label">${mode}</span>
        <div class="chart-bar-wrap">
          <div class="chart-bar" style="width: ${pct}%"></div>
        </div>
        <span class="chart-value">${formatMetric(value)}</span>
      </div>
    `;
  }).join("");

  requestAnimationFrame(() => {
    chartEl.querySelectorAll(".chart-bar").forEach((bar) => {
      const wrap = bar.parentElement;
      if (!wrap) return;
      const target = bar.style.width;
      bar.style.width = "0";
      requestAnimationFrame(() => {
        bar.style.width = target;
      });
    });
  });
}

/**
 * @param {object} data
 */
function renderPairwise(data) {
  const grid = document.getElementById("pairwise-grid");
  if (!grid || !Array.isArray(data.pairwise)) return;

  grid.innerHTML = data.pairwise
    .map((p) => {
      const title = `${p.left_mode} vs ${p.right_mode}`;
      return `
        <article class="pairwise-card">
          <h4>${title}</h4>
          <div class="pairwise-stats">
            <div class="pairwise-stat">
              <span class="pairwise-stat-value">${p.wins}</span>
              <span class="pairwise-stat-label">wins</span>
            </div>
            <div class="pairwise-stat">
              <span class="pairwise-stat-value">${p.ties}</span>
              <span class="pairwise-stat-label">ties</span>
            </div>
            <div class="pairwise-stat">
              <span class="pairwise-stat-value">${p.losses}</span>
              <span class="pairwise-stat-label">losses</span>
            </div>
          </div>
        </article>
      `;
    })
    .join("");
}

/**
 * @param {object} data
 */
function renderBenchmarkHash(data) {
  const hashEl = document.getElementById("benchmark-hash");
  const copyBtn = document.getElementById("hash-copy");
  if (!hashEl || !data.benchmark_sha256) return;

  const full = data.benchmark_sha256;
  const short = `${full.slice(0, 8)}…${full.slice(-8)}`;
  hashEl.textContent = short;
  hashEl.title = full;

  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(full);
        copyBtn.textContent = "Copied";
        setTimeout(() => {
          copyBtn.textContent = "Copy";
        }, 2000);
      } catch {
        copyBtn.textContent = "Copy failed";
      }
    });
  }
}

/**
 * @param {object} data
 */
function renderIncremental(data) {
  const grid = document.getElementById("incremental-grid");
  if (!grid || !Array.isArray(data.scenarios)) return;

  grid.innerHTML = data.scenarios
    .map((scenario) => {
      const label = SCENARIO_LABELS[scenario.id] || scenario.id;
      const fullAnalyzed = scenario.full.files_analyzed;
      const incAnalyzed = scenario.incremental.files_analyzed;
      const reused = scenario.dense_selective.vectors_reused;
      const embedded = scenario.dense_selective.vectors_embedded;
      const total = scenario.dense_selective.document_count;
      const reusedPct = total > 0 ? (reused / total) * 100 : 0;
      const embeddedPct = total > 0 ? (embedded / total) * 100 : 0;

      return `
        <article class="incremental-card">
          <h4>${label}</h4>
          <p class="incremental-desc">${scenario.description}</p>
          <div class="incremental-compare">
            <span class="full">Full: ${fullAnalyzed} files analyzed</span>
            <span class="arrow">→</span>
            <span class="inc">Incremental: ${incAnalyzed}</span>
          </div>
          <div class="vector-bar">
            <div class="vector-bar-label">
              <span>Vectors reused: ${reused}</span>
              <span>Embedded: ${embedded}</span>
            </div>
            <div class="vector-bar-track" role="img" aria-label="${reused} reused, ${embedded} embedded of ${total}">
              <div class="vector-bar-reused" style="width: ${reusedPct}%"></div>
              <div class="vector-bar-embedded" style="width: ${embeddedPct}%"></div>
            </div>
          </div>
        </article>
      `;
    })
    .join("");
}

/**
 * @param {object} data
 */
function initBenchmark(data) {
  benchmarkData = data;
  renderRetrievalShowcase(data);
  renderBenchmarkChart(data, activeMetric);
  renderPairwise(data);
  renderBenchmarkHash(data);
}

function setupMetricTabs() {
  const tabs = document.querySelectorAll(".metric-tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const metric = tab.getAttribute("data-metric");
      if (!metric || !benchmarkData) return;

      activeMetric = metric;
      tabs.forEach((t) => {
        const selected = t === tab;
        t.classList.toggle("active", selected);
        t.setAttribute("aria-selected", selected ? "true" : "false");
      });
      renderBenchmarkChart(benchmarkData, activeMetric);
    });
  });
}

function setupNavHighlight() {
  const sections = document.querySelectorAll("section[id]");
  const links = document.querySelectorAll(".nav-links a[href^='#']");

  if (!sections.length || !links.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const id = entry.target.getAttribute("id");
        links.forEach((link) => {
          const href = link.getAttribute("href");
          link.style.color =
            href === `#${id}` ? "var(--text)" : "";
        });
      });
    },
    { rootMargin: "-40% 0px -50% 0px", threshold: 0 }
  );

  sections.forEach((section) => observer.observe(section));
}

async function init() {
  setupMetricTabs();
  setupNavHighlight();

  try {
    const [benchmark, incremental] = await Promise.all([
      fetchJson(BENCHMARK_URL),
      fetchJson(INCREMENTAL_URL),
    ]);
    initBenchmark(benchmark);
    renderIncremental(incremental);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    showError(
      "benchmark-error",
      "Run the demo from the repository root with:"
    );
    showError(
      "incremental-error",
      "Run the demo from the repository root with:"
    );
    console.error("Dashboard data load failed:", msg);
  }
}

document.addEventListener("DOMContentLoaded", init);
