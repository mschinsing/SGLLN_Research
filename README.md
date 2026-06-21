# Semantically-Grounded Lead-Lag Networks

Directed lead-lag network analysis of Polymarket prediction markets over the 2024
US election cycle (Jun 1 – Nov 5, 2024). The pipeline discovers cross-market
information-flow structure, grounds it against contract semantics, and tests
whether it is exploitable.

**Headline finding:** prediction-market lead-lag structure is **real, robust, and
event-responsive but informationally efficient — detectable, not exploitable**,
with one strong positive sub-result: a significant **bottom-up information cascade
within politics** (granular state/congressional markets lead aggregate
electoral-margin markets, p < 0.001).

See **[`FINDINGS.md`](FINDINGS.md)** for the full results narrative and
**[`HYPOTHESES.md`](HYPOTHESES.md)** for the H1–H4 adjudication.
Architecture: `data/processed/presentation/00_architecture.png`.

---

## Key results

| Result | Number |
|---|---|
| Directed structure significance (top-10 eigenvalues) | p = 0.001 |
| Lead-lag clusters vs platform categories (ARI) | **0.04** (orthogonal to topic) |
| Semantic clusters vs categories (ARI) | 0.37 (embeddings validated) |
| Robust to liquidity (leadingness ~ log volume, R²) | 0.006 |
| Robust to common factor (directed edges surviving PC1 removal) | **80.7%** |
| Out-of-sample forecasting (directional accuracy) | ≈48–49% (null; 0/27 edges past FDR) |
| Break-even round-trip spread | ≈ 0.02¢ (vs realistic ≥1¢) |
| Within-politics cascade (congress → electoral/margin) | **p < 0.001** |

---

## Repository structure

```
config.py                 # ALL parameters & file paths (single source of truth)
utils.py                  # logging, HTTP, JSON/parquet IO, bootstrap helpers
run_all.py                # full reproduction driver (Phase 0→5 + reporting)

# Phase 0 — data acquisition & preprocessing
run_phase0.py  step0_1_fetch_markets.py … step0_6_supplementary.py

# Phase 1 — statistical lead-lag discovery
run_phase1.py  step1_1_leadlag.py … step1_6_liquidity.py

# Phase 2 — semantic grounding
run_phase2.py  step2_1_semantic_embedding.py … step2_5_edge_filter.py

# Phase 3 — walk-forward forecasting & validation
run_phase3.py  step3_1_walkforward.py … step3_4_event_analysis.py

# Robustness, synthesis, extensions, reporting
check_common_factor.py    # common-factor (PC1) robustness pre-check
phase4_ablations.py       # ablation / robustness studies
phase5_synthesis.py       # H1–H4 hypothesis adjudication
common_factor_figures.py  # edge-survival + before/after figures
politics_deepdive.py      # within-politics sub-topic meta-flow
make_report_figures.py    # presentation figure deck (01–12)
make_architecture_diagram.py

data/
  raw/        # fetched markets, prices, trades
  processed/  # returns, adjacency, clusters, forecasts, metrics … + presentation/ figures
FINDINGS.md  HYPOTHESES.md   # results & hypothesis adjudication
```

---

## Installation

```bash
git clone https://github.com/mschinsing/SGLLN_Research.git && cd SGLLN_Research
python -m venv venv && source venv/bin/activate     # Python 3.10
pip install -r requirements.txt
```

> **Note:** `threadpoolctl >= 3.5` is required — older 2.x versions crash
> scikit-learn KMeans on some conda/MKL builds.

---

## Reproduce everything

```bash
python run_all.py                 # full pipeline incl. ~1h Phase 0 data fetch
python run_all.py --skip-data     # skip the fetch; reuse data/processed/*
python run_all.py --skip-trades   # fetch, but skip the slow, unused trades step
```

Or run phases individually (each supports `--from N` / `--only N`):

```bash
python run_phase0.py              # data (slowest: step 0.2 price fetch ≈ 1h)
python run_phase1.py              # lead-lag network  (~7 min)
python run_phase2.py              # semantic grounding (~10 s)
python run_phase3.py              # walk-forward forecasting (~4 min)
python check_common_factor.py     # common-factor robustness
python phase4_ablations.py        # ablations
python phase5_synthesis.py        # hypothesis adjudication
python common_factor_figures.py && python politics_deepdive.py
python make_report_figures.py && python make_architecture_diagram.py
```

---

## Data provenance

| Source | Endpoint | Used for |
|---|---|---|
| Polymarket **Gamma** | `gamma-api.polymarket.com/markets` | market metadata, categories (`include_tag=true`), volume |
| Polymarket **CLOB** | `clob.polymarket.com/prices-history` | daily price history (`fidelity=1440`) |
| Polymarket **Data API** | `data-api.polymarket.com/trades` | trade liquidity (optional; not used downstream) |

Public read endpoints; no auth. Window: 2024-06-01 → 2024-11-05.
**Note:** intraday price history is *not retained* for these now-closed markets
(verified: `fidelity=60` returns empty), so the analysis is daily-resolution.

> **What ships in this repo:** the analysis-ready `data/processed/` is committed,
> so `python run_all.py --skip-data` reproduces Phases 1–5 out of the box. The
> 86 MB raw fetch (`data/raw/`, 4,434 JSONs) is **not** committed — regenerate it
> with `python run_phase0.py` to reproduce from scratch.

---

## Reproducibility notes

- **All parameters live in `config.py`** — dates, thresholds, k, lags, bootstrap
  sizes, paths. Nothing is hard-coded in the step files.
- `RANDOM_SEED = 42` fixes every stochastic step (bootstrap, k-means,
  permutation tests, walk-forward RNG).
- **One non-determinism caveat:** the Phase 0 market fetch orders by *live*
  cumulative volume, so the eligible-contract count can drift ±1 between fetches
  (e.g. 354 vs 355). Re-running Phases 1–5 on a fixed `data/processed/` is fully
  deterministic.
- Outputs are written as `.parquet` / `.json` / `.npy`; figures as `.png` in
  `data/processed/presentation/`.

---

## Figure index (`data/processed/presentation/`)

`00_architecture` · `01_dataset_overview` · `02_leadlag_significance` ·
`03_three_way_comparison` (centerpiece) · `04_cluster_structure` ·
`05_liquidity_controls` · `06_common_factor_check` · `07_event_conditional` ·
`08_ablations` · `09_category_metaflow` · `10_hypothesis_scorecard` ·
`11_common_factor_strength` · `12_politics_subflow`
