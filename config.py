"""
config.py
=========
Central configuration for the Phase 0 Polymarket pipeline.

Every constant the step files import lives here. Names and TYPES are chosen
to match exactly how the steps use them — in particular:

  * START_DATE / END_DATE are datetime.date objects (step0_4 compares them
    against parsed .date() values; step0_5 does `d += timedelta(days=1)`).
  * CLOB_URL and DATA_URL are FULL endpoint URLs, because the steps pass them
    straight into fetch_json(url, params=...).
  * All *_DIR / *_FILE paths are absolute, anchored to this file's folder, so
    the project works no matter what directory you launch it from.

Layout assumed (flat project root, matching your inventory):

    project/
    ├── config.py  utils.py  step0_1..step0_6  run_phase0.py
    └── data/
        ├── raw/{prices,trades}/
        └── processed/embeddings/
"""

import os
from datetime import date

# --------------------------------------------------------------------------
# Paths — anchored to this file's directory (the project root)
# --------------------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
RAW_DIR        = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR  = os.path.join(DATA_DIR, "processed")

PRICES_RAW_DIR = os.path.join(RAW_DIR, "prices")
TRADES_RAW_DIR = os.path.join(RAW_DIR, "trades")
EMBEDDINGS_DIR = os.path.join(PROCESSED_DIR, "embeddings")

# Raw outputs
MARKETS_RAW_FILE = os.path.join(RAW_DIR, "markets_raw.json")   # step 0.1
VOLUME_FILE      = os.path.join(RAW_DIR, "daily_volumes.parquet")  # step 0.3

# Processed outputs
FILTER_FUNNEL_FILE    = os.path.join(PROCESSED_DIR, "filter_funnel.csv")
ELIGIBLE_MARKETS_FILE = os.path.join(PROCESSED_DIR, "eligible_markets.json")
ELIGIBLE_IDS_FILE     = os.path.join(PROCESSED_DIR, "eligible_token_ids.json")
LOGIT_PRICES_FILE     = os.path.join(PROCESSED_DIR, "logit_prices.parquet")
RETURNS_FILE          = os.path.join(PROCESSED_DIR, "returns.parquet")
OVERLAP_MATRIX_FILE   = os.path.join(PROCESSED_DIR, "overlap_matrix.parquet")
LIQUIDITY_FILE        = os.path.join(PROCESSED_DIR, "liquidity_metadata.parquet")
EVENTS_FILE           = os.path.join(PROCESSED_DIR, "event_calendar.csv")
DESCRIPTIONS_FILE     = os.path.join(PROCESSED_DIR, "contract_descriptions.json")

# --------------------------------------------------------------------------
# Polymarket API endpoints  (verified June 2026; read paths are public)
# These are FULL endpoint URLs because the steps call fetch_json(URL, params).
# --------------------------------------------------------------------------
GAMMA_URL = "https://gamma-api.polymarket.com/markets"     # step 0.1 (markets discovery)
CLOB_URL  = "https://clob.polymarket.com/prices-history"   # step 0.2 (daily price history)
DATA_URL  = "https://data-api.polymarket.com/trades"       # step 0.3 (trade records)

# --------------------------------------------------------------------------
# HTTP / rate limiting (Gamma is ~60 req/min unauthenticated — keep margin)
# --------------------------------------------------------------------------
REQUEST_DELAY   = 0.5    # seconds to sleep after a successful request
REQUEST_TIMEOUT = 30     # per-request timeout (s)
MAX_RETRIES     = 4      # attempts before fetch_json raises RuntimeError
RETRY_BACKOFF   = 1.5    # exponential backoff base: wait = backoff ** attempt

# --------------------------------------------------------------------------
# Price-history params (step 0.2)
# NOTE: on the CLOB prices-history endpoint, `fidelity` is the resolution in
# MINUTES (1440 = one point per day), and `interval` is the lookback window
# ("max" = full history). Your inline comments labelled these differently;
# these values are what the live endpoint actually expects for daily bars.
# --------------------------------------------------------------------------
PRICE_INTERVAL = "max"
PRICE_FIDELITY = 1440

# --------------------------------------------------------------------------
# INTRADAY re-analysis (parallel track — never overwrites the daily pipeline)
# The /prices-history endpoint returns empty for interval=max&fidelity<1440, but
# returns full intraday data when queried with explicit startTs/endTs in <=15-day
# chunks. Hourly (fidelity=60) for the network; minute (fidelity=1) for spreads.
# --------------------------------------------------------------------------
PRICES_HOURLY_DIR   = os.path.join(RAW_DIR, "prices_hourly")
INTRADAY_DIR        = os.path.join(PROCESSED_DIR, "intraday")
RETURNS_HOURLY_FILE = os.path.join(INTRADAY_DIR, "returns_hourly.parquet")
INTRADAY_SUMMARY_FILE = os.path.join(INTRADAY_DIR, "intraday_summary.json")
INTRADAY_COSTS_FILE   = os.path.join(INTRADAY_DIR, "intraday_costs.json")
INTRADAY_FIDELITY   = 60     # minutes per bar (60 = hourly)
INTRADAY_CHUNK_DAYS = 15     # API returns empty for longer ranges → chunk
MAX_LAG_HOURS       = 24     # lead-lag lags 1..24 hours (spans the daily lag-1)

# --------------------------------------------------------------------------
# Study window: 2024 US election cycle.  date objects (NOT strings) — required.
# --------------------------------------------------------------------------
START_DATE = date(2024, 6, 1)
END_DATE   = date(2024, 11, 5)                     # election day; plan window
TOTAL_DAYS = (END_DATE - START_DATE).days + 1     # 158

# --------------------------------------------------------------------------
# Step 0.4 — eligibility filter thresholds (funnel labels reference these)
# --------------------------------------------------------------------------
MIN_ACTIVE_DAYS     = 60          # "2_active_days_ge_60"
MIN_VOLUME_USD      = 50_000      # "3_volume_ge_50k"
MIN_RETURN_STD      = 0.01        # "4_return_std_gt_001"
MAX_GAP_DAYS        = 5           # "5_max_gap_le_5"
RESOLUTION_EXCLUDE_H = 48         # drop obs in the final N hours before resolution

# --------------------------------------------------------------------------
# Step 0.4 / 0.5 — logit clipping and winsorization
# --------------------------------------------------------------------------
CLIP_LO = 0.02      # clip prices into [CLIP_LO, CLIP_HI] before logit
CLIP_HI = 0.98
WINSOR_LO = 0.01    # winsorize returns to [1st, 99th] pct (passed as fractions)
WINSOR_HI = 0.99

# --------------------------------------------------------------------------
# Step 0.5 — pairwise overlap requirement
# --------------------------------------------------------------------------
MIN_OVERLAP_DAYS = 30

# --------------------------------------------------------------------------
# step 0.1 fetch tuning (used by step0_1_fetch_markets.py)
# --------------------------------------------------------------------------
GAMMA_PAGE_LIMIT = 100     # markets per page; Gamma caps `limit` at 100 server-side
GAMMA_MAX_PAGES  = 200     # safety cap on pagination
GAMMA_OFFSET_MAX = 10_000  # Gamma rejects offset > 10000 with HTTP 422
GAMMA_VOLUME_MIN = 0       # optional server-side volume floor for discovery

# --------------------------------------------------------------------------
# Category normalization (applied in step 0.1 derive_category and step 0.4)
# Folds fallback-tag stragglers into their canonical top-level category so the
# cluster-vs-category validation has a clean 5-8 category ground truth.
# --------------------------------------------------------------------------
CATEGORY_ALIASES = {
    "Crypto Prices":      "Crypto",
    "Solana":             "Crypto",
    "US-current-affairs": "Politics",
    "Celebrities":        "Entertainment",
}

# Macro/monetary-policy markets get tagged "Politics" by Gamma but are
# economically (not electorally) driven. Reclassify by question text so the
# Fed-date event study and the cluster-vs-category validation are clean.
MACRO_CATEGORY = "Economy"
MACRO_KEYWORDS = (
    "fed ", "interest rate", "rate cut", "rate hike", "fomc",
    "government shutdown", "recession", "jobs report", "nonfarm",
    "inflation", "cpi", "gdp",
)

RANDOM_SEED = 42

# ==========================================================================
# PHASE 1 — Statistical Lead–Lag Discovery
# ==========================================================================
# Inputs are the Phase 0 processed outputs: RETURNS_FILE, OVERLAP_MATRIX_FILE,
# LIQUIDITY_FILE, ELIGIBLE_MARKETS_FILE (all defined above).

PHASE1_DIR  = os.path.join(PROCESSED_DIR, "phase1")
FIGURES_DIR = os.path.join(PHASE1_DIR, "figures")

# Outputs
LEADLAG_SCORES_FILE     = os.path.join(PHASE1_DIR, "leadlag_scores.parquet")   # step 1.1
IDCOR_MATRIX_FILE       = os.path.join(PHASE1_DIR, "idcor_matrix.parquet")     # step 1.1 (dense I)
NULL_DIST_FILE          = os.path.join(PHASE1_DIR, "null_distribution.npy")    # step 1.2
TAU_DEP_FILE            = os.path.join(PHASE1_DIR, "tau_dep.json")             # step 1.2
ADJACENCY_FILE          = os.path.join(PHASE1_DIR, "adjacency_matrix.parquet") # step 1.3
CLUSTERS_FILE           = os.path.join(PHASE1_DIR, "clusters.json")            # step 1.4
EIGEN_FILE              = os.path.join(PHASE1_DIR, "eigenvalues.npy")          # step 1.4
NODE_EMBEDDINGS_FILE    = os.path.join(PHASE1_DIR, "node_embeddings.npy")      # step 1.4
METAFLOW_FILE           = os.path.join(PHASE1_DIR, "meta_flow.parquet")        # step 1.5
CLUSTER_RANKING_FILE    = os.path.join(PHASE1_DIR, "cluster_ranking.csv")      # step 1.5
LIQUIDITY_CONTROLS_FILE = os.path.join(PHASE1_DIR, "liquidity_controls.json")  # step 1.6

# Step 1.1 — pairwise lead–lag metrics
MAX_LAG         = 5      # lags l = 1..MAX_LAG for the CCF-AUC integrals
PAIR_CHUNK_SIZE = 500    # pairs per joblib task (chunked to amortize dispatch overhead)
N_JOBS          = -1     # joblib workers (-1 = all cores)
# MIN_OVERLAP_DAYS (=30) from Phase 0 also gates the runtime per-pair overlap check.

# Step 1.2 — significance / dependence floor (single global tau_dep)
BOOTSTRAP_B      = 1000  # stationary-block-bootstrap replicates
NULL_SAMPLE_PAIRS = 1000 # random pairs sampled to build the pooled null
DEP_FLOOR_PCT    = 95    # percentile of pooled null max-I -> tau_dep

# Step 1.4 — Hermitian spectral clustering
K_VALUES = [3, 5, 7, 10] # cluster-count sensitivity sweep (analyst's robustness range)
EIGENGAP_KMIN = 2        # real floor the eigengap may select (k=1 = no clustering);
                         # decoupled from K_VALUES so the heuristic is data-driven
PERM_B   = 1000          # permutation reps for eigenvalue significance
EMBED_ROW_NORMALIZE = True  # NJW-style L2 row-normalization of embeddings before k-means

# Step 1.6 — liquidity controls
N_VOLUME_TERCILES = 3

# ==========================================================================
# PHASE 2 — Semantic Grounding
# ==========================================================================
# Reuses Phase 1 outputs: CLUSTERS_FILE (statistical clusters + k_selected),
# NODE_EMBEDDINGS_FILE (v_i), METAFLOW_FILE, and Phase 0 ELIGIBLE_MARKETS_FILE.
PHASE2_DIR         = os.path.join(PROCESSED_DIR, "phase2")
PHASE2_FIGURES_DIR = os.path.join(PHASE2_DIR, "figures")

# Outputs
SEM_EMBEDDINGS_FILE        = os.path.join(PHASE2_DIR, "contract_embeddings.npy")        # step 2.1
SEM_SIM_FILE               = os.path.join(PHASE2_DIR, "semantic_similarity.parquet")    # step 2.1
SEM_CLUSTERS_FILE          = os.path.join(PHASE2_DIR, "semantic_clusters.json")         # step 2.2
THREEWAY_FILE              = os.path.join(PHASE2_DIR, "three_way_comparison.json")       # step 2.3
FUSED_CLUSTERS_FILE        = os.path.join(PHASE2_DIR, "fused_clusters.json")            # step 2.4
SEM_FILTERED_METAFLOW_FILE = os.path.join(PHASE2_DIR, "semantic_filtered_metaflow.parquet")  # step 2.5

# Sentence-BERT model. IMPORTANT: Phase 2 embeds the contract QUESTION TEXT ONLY
# (no category/outcome suffix), so the semantic signal is independent of the
# platform categories it is later compared against in the three-way analysis.
SEM_MODEL     = "all-MiniLM-L6-v2"   # 384-dim, CPU-friendly
SCS_NULL_B    = 1000                 # random-clustering reps for SCS z-scores (step 2.3)
FUSION_ALPHAS = [0.25, 0.5, 1.0, 2.0]  # semantic weighting sweep for fusion (step 2.4)
SPS_PERCENTILE = 50                  # keep meta-flow edges with SPS above this pct (step 2.5)

# ==========================================================================
# PHASE 3 — Walk-Forward Forecasting & Validation
# ==========================================================================
# Reuses Phase 1/2 functions; re-estimates everything per window (no lookahead).
PHASE3_DIR = os.path.join(PROCESSED_DIR, "phase3")

# Inputs: LOGIT_PRICES_FILE (un-winsorized → re-winsorize per window, avoids leak),
# SEM_SIM_FILE (static semantic similarity), CLUSTERS_FILE, EVENTS_FILE.

# Outputs
FORECASTS_FILE        = os.path.join(PHASE3_DIR, "forecasts.parquet")          # 3.1-3.3
WF_METRICS_FILE       = os.path.join(PHASE3_DIR, "forecast_metrics.json")       # 3.4
CALIBRATION_FILE      = os.path.join(PHASE3_DIR, "calibration.json")            # 3.5
EVENT_ANALYSIS_FILE   = os.path.join(PHASE3_DIR, "event_analysis.json")         # 3.6

# Walk-forward framework (3.1)
W_TRAIN          = 60     # training window length (days)
WF_STEP          = 5      # forecast every WF_STEP days (compute lever; daily=1)
WF_K             = 7      # fixed cluster count across windows (comparability)
WF_DEP_PCT       = 86     # within-window adjacency gate: keep top ~14% of pairs by max-I
                          # (fast per-window proxy for the τ_dep bootstrap floor)
WF_MIN_ACTIVE    = 30     # contract must have >= this many non-NaN returns in-window to trade
WF_REMOVE_COMMON_FACTOR = [False, True]  # run both: with/without PC1 residualization (#2)
WF_VARIANTS      = ["unfiltered", "filtered", "random"]  # three signal variants (3.3)

# Evaluation (3.4)
N_BOOTSTRAP_CI   = 1000   # bootstrap replicates for metric confidence intervals

# ==========================================================================
# PHASE 4 — Ablation / robustness studies (phase4_ablations.py)
# ==========================================================================
PHASE4_DIR               = os.path.join(PROCESSED_DIR, "phase4")
ABLATION_SUMMARY_CSV     = os.path.join(PHASE4_DIR, "ablation_summary.csv")
ABLATION_SUMMARY_JSON    = os.path.join(PHASE4_DIR, "ablation_summary.json")
COMMON_FACTOR_CHECK_FILE = os.path.join(PHASE1_DIR, "common_factor_check.json")

# ==========================================================================
# PHASE 5 — Synthesis & reporting (phase5_synthesis.py)
# ==========================================================================
PHASE5_DIR        = os.path.join(PROCESSED_DIR, "phase5")
HYPOTHESIS_JSON   = os.path.join(PHASE5_DIR, "hypothesis_adjudication.json")
HYPOTHESES_MD     = os.path.join(BASE_DIR, "HYPOTHESES.md")   # project-root summary

# Paper-strengthening extensions
COMMON_FACTOR_RESID_ADJ_FILE = os.path.join(PHASE1_DIR, "common_factor_resid_adjacency.parquet")
COMMON_FACTOR_STRENGTH_FILE  = os.path.join(PHASE1_DIR, "common_factor_strength.json")
POLITICS_SUBFLOW_FILE        = os.path.join(PHASE5_DIR, "politics_subflow.json")

# Reporting — presentation-ready figure deck (make_report_figures.py)
PRESENTATION_DIR = os.path.join(PROCESSED_DIR, "presentation")
