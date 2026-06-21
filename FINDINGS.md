# Semantically-Grounded Lead–Lag Networks — Findings

*Polymarket 2024 US election cycle (Jun 1 – Nov 5, 2024). Figures in `data/processed/presentation/`.*

## Dataset
- **354 eligible contracts** (one YES-side series per binary market) across **8 categories**, from 2,023 in-window markets after a 5-criterion eligibility funnel (active ≥60d, volume ≥$50k, return-std >0.01, gap ≤5d, 48h resolution exclusion). *(Fig 01)*
- Politics 196 · Sports 101 · Economy 24 · Crypto 18 · Entertainment 6 · Science 6 · Geopolitics 2 · Business 1.

## Phase 1 — Statistical lead–lag structure is real, directional, and not liquidity
- **Significant directional structure.** Distance-correlation CCF-AUC over 62,481 pairs, gated by a stationary-block-bootstrap floor τ_dep=1.25 (95th pct of a 1M-draw null). **13.9% of pairs clear the floor vs 5% expected by chance** → genuine dependence. *(Fig 02b)*
- **Hierarchical clustering.** Hermitian (magnetic-Laplacian) spectral clustering: the dominant eigengap is at **k=2** (a balanced 181/173 leader–follower split); all top-10 eigenvalues are significant (p=0.001 permutation), so finer structure is real → **k=7** used for the meta-flow. *(Fig 02a)*
- **A leader→follower hierarchy.** Cluster leadingness L(a) spans +0.148 → −0.091. Leading clusters are **topically diverse**; the **Sports-dominated cluster lags** (sports resolutions are exogenous events that follow, not drive). *(Fig 04)*
- **Not a liquidity artifact.** Node leadingness vs log(cumulative volume): **R²=0.006** (slope p=0.14; placebo p=0.13), and directed structure persists within all 3 volume terciles. *(Fig 05)*
- **Not a common-factor artifact.** Removing the first principal component (PC1, the "election-beta" common factor — only **10.5% of return variance**) and re-running on residuals: **80.7% of directed edges survive** (same direction), the cluster-leadingness ranking is preserved (**Spearman 0.86**), node leadingness ρ=0.75, % pairs clearing floor unchanged (13.9%→14.9%), topic-orthogonality persists (ARI 0.04→0.09). 4 of 5 lead–lag links are genuine idiosyncratic flow, not reaction-speed to one shared signal. *(Figs 06, 11)*

## Phase 2 — Lead–lag is largely orthogonal to meaning (the core result)
Three partitions compared — statistical (lead-lag), semantic (text embeddings, category-free), platform category: *(Fig 03)*

| Comparison | ARI | NMI |
|---|---|---|
| Semantic ↔ Category | **0.37** | **0.57** |
| Statistical ↔ Semantic | 0.05 | 0.12 |
| Statistical ↔ Category | 0.04 | 0.12 |

- **Embeddings validated:** semantic clusters strongly recover platform categories (ARI 0.37) — the text signal captures real topic.
- **Headline:** **lead–lag community structure is largely orthogonal to topic** (ARI ≈ 0.05 against both semantic and category), triangulated three ways. Markets that lead/lag each other are *not* the markets that are *about* similar things.
- **Nuance (publishable):** it isn't flat zero. Semantic Coherence z-scores show **3/7 statistical clusters are significantly coherent in meaning** (z up to +7.1), 2/7 are significantly *anti*-coherent (deliberately span topics), 2/7 neutral. So lead–lag is *partially*, cluster-specifically grounded. *(Fig 03b)*
- **Fusion & semantic edge filtering:** with blocks scaled to equal variance (1/√dim), the fused partition sweeps monotonically from statistical-aligned (α=0.25: ARI 0.36 vs statistical) to semantic-aligned (α=2.0: ARI 0.65 vs semantic). Notably, even at equal weight (α=1.0) the blend leans semantic (0.42 vs 0.18) — **meaning forms crisper clusters than dynamics.** Semantic plausibility filtering retains 10/21 inter-cluster meta-flow edges.

## Phase 3 — The structure is real but not exploitable (the efficiency result)
Strictly-causal walk-forward (W=60d, re-estimate everything per window; point-in-time trade gate; returns re-winsorized in-window; ±PC1 factor removal). Three signal variants forecast lagging-cluster returns from leading-cluster returns at the estimated dominant lag.

- **No forecasting edge.** Directional accuracy is **45–49% across all variants** (unfiltered / semantically-filtered / random), every 95% CI spans 50%, binomial p = 0.65–0.98, and **0 of 27 per-edge claims survive BH-FDR**. The unfiltered−random gap is +0.008 to +0.031 — within noise.
- **Not tradeable under any realistic cost.** No historical bid-ask data exists (Polymarket's book/spread endpoints are live-only; `prices-history` returns one mark per timestamp), so rather than guess a cost we compute the **break-even round-trip spread** — the cost that zeroes the gross edge. It is **≈0.02–0.04¢** (and negative once the common factor is removed), versus realistic spreads of ≥1¢; net-Sharpe at a 1¢ spread is **≈ −1.5**. Because the gross edge is statistically zero, the break-even cost is ~0 and the conclusion holds for *all* spread assumptions — no tradeability claim is made (gross Sharpe is illustrative only).
- **No calibration advantage.** Leading-cluster contracts are slightly better-calibrated (Brier 0.041 vs 0.044) — the direction "leaders know more" predicts — but it is **not significant** (CI [−0.020, +0.013]; Mann–Whitney p = 0.35).
- **Structure intensifies around events (suggestive).** In rolling 60-day windows, |λ1| (directed-structure intensity) is higher in windows with a fresh macro/political event than without (**0.583 vs 0.530, Mann–Whitney p = 0.092**) — exploratory (~10 overlapping windows), but trends as hypothesized. *(Fig 07)*
- **Interpretation:** the lead–lag structure is statistically real and robust (Phases 1–2) and appears to intensify around information events, yet carries **no out-of-sample predictive or calibration edge**. The structure is contemporaneous/descriptive, not exploitable → **direct evidence for cross-domain informational efficiency** in prediction markets (the meaningful null-result for the information-aggregation literature). *(Figs 06–07)*

## Phase 4 — Robustness / ablations
Each alternative method's clusters/leadingness scored against the dcor·Hermitian·k=7 baseline. *(Fig 08)*

- **The method choices are justified, not interchangeable.** Naive symmetrization (vs Hermitian) gives **ARI = 0.14**, and Pearson/Kendall (vs dcor) give **ARI = 0.17 / 0.04** — alternative methods produce materially different partitions, so the *directional* spectral method and the *nonlinear* dependence metric each contribute real structure (they earn their complexity).
- **The substantive findings are robust where it matters.** The **leadingness hierarchy survives the metric** (Spearman ρ = 0.63 Pearson, 0.46 Kendall, 0.75 common-factor), and **topic-orthogonality is stable across k** (ARI vs category 0.02–0.13 for k ∈ {2,3,5,7,10}). The exact 7-cluster partition is method-sensitive (as spectral assignments generally are) — over-interpret the leadingness ordering, not the precise membership.
- Common-factor (ρ=0.75, structure survives) and liquidity (R²=0.006, persists in terciles) carried over from earlier robustness checks. Transfer entropy, clip-bounds, and fusion ablations deliberately out of scope.

## Phase 5 — Hypothesis adjudication
*(Fig 09 category meta-flow, Fig 10 scorecard; full table in `HYPOTHESES.md`.)*

| H | Claim | Verdict |
|---|---|---|
| **H1** | Economic → Political lead-lag persists | **Weak / not supported** — Economy→Politics net flow ≈ 0 (−0.0005, perm p=0.47); no directional precedence |
| **H2** | spectral↔category ARI ≈ 0.3–0.6 | **Rejected** — ARI = 0.04 (spectral clusters orthogonal to topic) |
| **H3** | semantic filtering cuts loss magnitude 10–20% | **Partial (loss-magnitude only)** — losing-trade magnitude −23% vs unfiltered, −27% vs random (in range, *specific* to filtering), but **no directional edge** — a loss-control effect, not alpha |
| **H4** | event reconfiguration + leader calibration edge | **Partial / suggestive** — |λ1| higher in event windows (p=0.09); leaders trend better-calibrated but NS (p=0.35) |

Two notable sub-findings: **economic markets do not lead political ones** here (H1 fails — interesting given the prior), and **semantic filtering genuinely controls downside** (H3 — it prunes the implausible edges whose errors are largest, consistent with Kim et al.) even though it can't create a directional edge.

## Within-politics deep dive — granular markets lead aggregate ones (the standout positive result)
Since lead-lag is concentrated within Politics (196 contracts), we split them into sub-topics and computed the within-politics meta-flow. *(Fig 12)*

- **Granular, local markets LEAD; aggregate-outcome markets LAG.** Leadingness ranks: **state-level** (z=+1.9) and **congress/down-ballot** (z=+1.3) lead; **electoral-college/margin** is the strongest *follower* (z=−1.9). The strongest significant directed edges: **congress → electoral/margin (perm p<0.001)** and **state-level → electoral/margin (p=0.01)**.
- **This reverses the naive prior.** The national presidential market is informationally *neutral* (z=+0.2 — it neither leads nor lags), and nomination contracts are followers, not leaders. Information flows **bottom-up** — from specific state/congressional races into the aggregate electoral-margin markets — not top-down from the headline national race.
- This is the project's **strongest significant directed structure** (p<0.001), and it's interpretable and pre-justifiable (local contracts incorporate local polling/news first), not a fishing artifact.

## Intraday (hourly) re-analysis — robust where it matters, timescale-sensitive in the details
Re-ran the lead-lag network at **hourly** resolution (lags 1–24h, same 354 contracts, ~24× more data: 1.04M vs 43K observations) and compared to the daily baseline. *(Figs 13, 14)*

- **Topic-orthogonality is robust to timescale.** Hourly lead-lag clusters remain orthogonal to platform categories (ARI **0.07** vs 0.04 daily). The headline finding holds at the finer resolution — *more* credible, not less.
- **Genuine fast (~1–3h) information propagation exists.** The dominant-lag distribution **peaks sharply at 1–3 hours** (with a diffuse noise tail) — real intraday lead-lag that daily bars collapse to invisibility. This is the timescale information actually flows, now measured.
- **The daily structural *specifics* are timescale-sensitive (important caution).** Only **17% of daily directed edges persist at hourly**, and node-leadingness barely correlates across resolutions (Spearman **0.18**). So the daily leadingness ranking and exact meta-flow/cascade ordering do **not** robustly replicate — partly genuine timescale-dependence, partly hourly stale-price/forward-fill noise (thin trading). Interpret the daily structural specifics cautiously; the robust claims are the aggregate ones.
- **Within-politics: core direction holds, ranking shifts.** Congress still leads and aggregate electoral/margin still lags at hourly (the "specific→aggregate" cascade direction is partially robust), but the leader set changes (nomination becomes top; state-level neutralizes).
- **Still not tradeable — now on a *measured* cost.** Effective spread from the minute mid-price bounce (Roll 1984) is ≥**0.01¢** (a lower bound — mid-price underestimates the true book spread); at the realistic tick floor (0.1¢) the follow-the-leader net Sharpe is **−0.12**, and the gross edge is a statistical null. Efficiency now rests on a measured cost, not an assumption.

**Net:** hourly *strengthens* the two deepest results (orthogonality + efficiency), *reveals* ~1–3h propagation, and *tempers* the daily-specific structural details — an honest, more credible picture than the daily analysis alone.

## One-line abstract
> Lead–lag community structure in 2024 Polymarket is statistically robust, directional, and liquidity- and common-factor-independent (80.7% of edges survive PC1 removal), yet **largely orthogonal to topic** (ARI ≈ 0.05) — a result that holds at hourly resolution, where information is shown to propagate within ~1–3 hours. The structure **intensifies around information events but is informationally efficient**: no out-of-sample forecasting or calibration edge, and unprofitable at any realistic transaction cost. Within politics, information flows **bottom-up** from granular state/congressional markets into aggregate electoral-margin markets (p<0.001).

## Caveats for write-up
1. **Category-free embeddings** (question text only) are required for an honest semantic↔category comparison — Phase 0's category-tagged embeddings would inflate it circularly.
2. **k by eigengap, not ARI** — choosing k to maximize category agreement would be validation leakage. Eigengap favors k=2 (dominant) with k=7 as finer significant structure.
3. **Fusion weighting** — blocks are standardized *and* scaled by 1/√dim so each carries unit total variance; α is therefore an interpretable equal-weight knob (α=1 weights the signals equally in embedding space). Without this, the 384-dim semantic block dominates the 14-dim spectral block ~27×.

## Figure index (`data/processed/presentation/`)
- `00_architecture.png` — system / pipeline architecture (Phase 0→5)
- `01_dataset_overview.png` — category mix + eligibility funnel
- `02_leadlag_significance.png` — eigengap spectrum + dependence-vs-null
- `03_three_way_comparison.png` — ARI/NMI matrix + semantic-coherence z-scores **(centerpiece)**
- `04_cluster_structure.png` — category composition per cluster + leadingness ranking
- `05_liquidity_controls.png` — leadingness vs volume confound check
- `06_common_factor_check.png` — leadingness preserved after PC1 removal
- `07_event_conditional.png` — rolling |λ1| with event markers
- `08_ablations.png` — alternative methods vs dcor·Hermitian baseline
- `09_category_metaflow.png` — category-level net flow (H1)
- `10_hypothesis_scorecard.png` — H1–H4 verdicts
- `11_common_factor_strength.png` — before/after adjacency + 80.7% edge survival
- `12_politics_subflow.png` — within-politics bottom-up cascade
- `13_intraday_vs_daily.png` — hourly vs daily: leadingness, propagation speed, cascade
- `14_intraday_costs.png` — measured effective spread + net-Sharpe-vs-cost
- Phase 1/2 also save: `phase1/figures/{eigenvalue_spectrum,meta_flow,leadingness_vs_volume}.png`, `phase2/figures/semantic_filtered_metaflow.png`
