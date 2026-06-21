# Hypothesis Adjudication (H1–H4)

Mapping of all results to the four pre-registered hypotheses. See `FINDINGS.md` for the full narrative and `data/processed/presentation/` for figures.

| H | Claim | Verdict | Key evidence |
|---|---|---|---|
| H1 | Economic → Political lead-lag persists after liquidity controls | **WEAK / NOT SUPPORTED** | Economy→Politics net flow = -0.0005 (perm p=0.470); liquidity R²=0.006 (not liquidity-driven); common-factor leadingness ρ=0.75 |
| H2 | ARI(spectral clusters, categories) ≈ 0.3–0.6 | **REJECTED** | statistical↔category ARI=0.044 (far below 0.3). (Semantic↔category ARI=0.37 is in range, but those are text clusters, not spectral.) |
| H3 | Semantic filtering reduces loss magnitude by 10–20% | **PARTIAL (loss-magnitude only)** | avg losing-trade magnitude: unfiltered=0.0167 → filtered=0.0128 (+23% vs unfiltered, +27% vs random) — in/above the 10–20% range and specific to filtering. BUT no directional edge (unfiltered−random CI [-0.056,+0.075] spans 0): a loss-control effect, not alpha. |
| H4 | Event-driven reconfiguration AND calibration advantage for leaders | **PARTIAL / SUGGESTIVE** | |λ1| event 0.583 vs non-event 0.530 (MW p=0.092, exploratory); calibration leaders Brier 0.041 vs 0.044 (p=0.35, NS) |

## Headline
Prediction-market lead-lag structure is **real, robust, and event-responsive but informationally efficient** — detectable yet not exploitable. H2 is a clean rejection (spectral clusters are orthogonal to topic); H3 fails (no forecast edge); H1 and H4 are directional/partial. The null forecasting result (Phase 3) is the substantive finding for the information-aggregation literature (Step 5.3).

## Step 5.2 — Deliverable index
| Deliverable | Artifact |
|---|---|
| Filtering funnel | `data/processed/filter_funnel.csv`; Fig `01_dataset_overview.png` |
| Meta-flow graph (full) | `phase1/figures/meta_flow.png` |
| Meta-flow (event-conditional, rolling) | Fig `07_event_conditional.png` |
| Category meta-flow (H1) | Fig `09_category_metaflow.png` |
| Three-way ARI/NMI matrix | Fig `03_three_way_comparison.png`; `phase2/three_way_comparison.json` |
| Forecasting comparison (unfiltered/filtered/random) | `phase3/forecast_metrics.json` |
| Leadingness vs volume | Fig `05_liquidity_controls.png` |
| Rolling eigenvalue / leadingness | Fig `07_event_conditional.png` |
| Ablation summary | `phase4/ablation_summary.csv`; Fig `08_ablations.png` |
| Hypothesis scorecard | Fig `10_hypothesis_scorecard.png` |
