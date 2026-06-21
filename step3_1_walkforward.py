"""
Step 3.1–3.3 — Walk-forward forecasting engine
==============================================
Strictly causal walk-forward. For each forecast date t (stepped by WF_STEP), the
ENTIRE pipeline is re-estimated using only the training window [t-W, t-1]:
returns (re-winsorized in-window), optional PC1 residualization, lead-lag
adjacency (within-window dependence gate), Hermitian clusters (fixed k), meta-flow.

Leakage controls:
  * Returns rebuilt from UN-winsorized logit prices, re-winsorized per window (#3).
  * Point-in-time active gate: only contracts with >= WF_MIN_ACTIVE in-window
    returns AND a value on day t are tradeable (#1 survivorship).
  * Lag, regression coefficients, clusters, SPS — all estimated on [t-W, t-1] only.
  * Optional common-factor (PC1) removal as a with/without variant (#2).

Lag-aligned signal (3.2): for each leading→lagging cluster edge a→b, pick the
dominant lag l* = argmax_l |dcor(r̄^Ca_{·-l}, r̄^Cb_·)| on the window, fit a signed
OLS r̄^Cb ~ r̄^Ca_{-l*}, and forecast r̄^Cb_t from the (known) r̄^Ca_{t-l*}.

Three variants (3.3): unfiltered (all edges) · filtered (SPS-passing edges) ·
random (permuted cluster labels → structure destroyed).

Input:  data/processed/logit_prices.parquet, data/processed/phase2/semantic_similarity.parquet
Output: data/processed/phase3/forecasts.parquet
        columns: date, factor_removed, variant, lead_cluster, lag_cluster, lag,
                 forecast, realized, n_lead, n_lag
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from joblib import Parallel, delayed

from config import (
    LOGIT_PRICES_FILE, SEM_SIM_FILE, FORECASTS_FILE, PHASE3_DIR,
    W_TRAIN, WF_STEP, WF_K, WF_DEP_PCT, WF_MIN_ACTIVE,
    WF_REMOVE_COMMON_FACTOR, MAX_LAG, MIN_OVERLAP_DAYS, SPS_PERCENTILE,
    WINSOR_LO, WINSOR_HI, N_JOBS, RANDOM_SEED,
)
from utils import get_logger, ensure_dirs
from step1_1_leadlag import _directional_integral, _abs_dcor
from step1_4_spectral import _normalized_hermitian, _embeddings, _cluster
from step1_5_metaflow import _meta_flow
from step1_6_liquidity import _build_adjacency
from step2_5_edge_filter import _sps_matrix
from step0_5_preprocess import winsorize

log = get_logger("step3.1_walkforward")


# --------------------------------------------------------------------------
def _residualize_pc1(R):
    """Remove the first principal component (common factor) from window returns."""
    filled = np.nan_to_num(R, nan=0.0)
    centered = filled - filled.mean(axis=0, keepdims=True)
    U, s, _ = np.linalg.svd(centered, full_matrices=False)
    f = U[:, 0] * s[0]
    out = np.full_like(R, np.nan)
    design = np.column_stack([np.ones_like(f), f])
    for j in range(R.shape[1]):
        col = R[:, j]
        m = ~np.isnan(col)
        if m.sum() < 3:
            continue
        beta, *_ = np.linalg.lstsq(design[m], col[m], rcond=None)
        out[m, j] = col[m] - design[m] @ beta
    return out


def _idcor_window(R):
    """Directed dcor integral matrix for an in-window return block (W × m)."""
    m = R.shape[1]
    idc = np.zeros((m, m))
    for i in range(m):
        ri = R[:, i]
        for j in range(i + 1, m):
            rj = R[:, j]
            co = (~np.isnan(ri)) & (~np.isnan(rj))
            if co.sum() < MIN_OVERLAP_DAYS:
                continue
            idc[i, j] = _directional_integral(ri, rj, MAX_LAG, _abs_dcor)
            idc[j, i] = _directional_integral(rj, ri, MAX_LAG, _abs_dcor)
    return idc


def _cluster_window(adj, k):
    a_sym, _, _ = _normalized_hermitian(adj)
    eigvals, eigvecs = eigh(a_sym)
    emb = _embeddings(eigvals, eigvecs, k, row_normalize=True)
    return _cluster(emb, k)


def _cluster_means(R, labels, k):
    """Per-day cluster-mean return series (k × W); NaN where a cluster has no data that day."""
    means = np.full((k, R.shape[0]), np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN day → NaN
        for a in range(k):
            cols = np.nonzero(labels == a)[0]
            if len(cols):
                means[a] = np.nanmean(R[:, cols], axis=1)
    return means


def _realized(rr, t_idx, active, labels):
    """Cluster-mean realized return on the forecast day t (out of sample)."""
    real = np.full(WF_K, np.nan)
    for a in range(WF_K):
        cols = active[labels == a]
        if len(cols):
            real[a] = np.nanmean(rr[t_idx, cols])
    return real


def _variant_rows(variant, adj, labels, means, real, rng, ctx):
    """Forecast rows for one signal variant at one window."""
    if variant == "random":                              # destroy structure↔membership link
        labels = rng.permutation(labels)
        means = _cluster_means(ctx["R"], labels, WF_K)
        real = _realized(ctx["rr"], ctx["t_idx"], ctx["active"], labels)
    flow = _meta_flow(adj, labels, WF_K)
    sps, thr = None, None
    if variant == "filtered":
        sps = _sps_matrix(ctx["sem_active"], labels, WF_K)
        thr = np.percentile(sps[np.triu_indices(WF_K, 1)], SPS_PERCENTILE)

    rows = []
    for a in range(WF_K):
        for b in range(WF_K):
            if a == b or flow[a, b] <= 0:                # require a leads b
                continue
            if thr is not None and sps[a, b] <= thr:     # semantic gate
                continue
            fc = _edge_forecast_with_lead(means[a], means[b], real[b])
            if fc is None:
                continue
            lag, forecast = fc
            rows.append({
                "date": ctx["date"], "factor_removed": ctx["remove_factor"], "variant": variant,
                "lead_cluster": a, "lag_cluster": b, "lag": lag,
                "forecast": forecast, "realized": float(real[b]),
                "n_lead": int((labels == a).sum()), "n_lag": int((labels == b).sum()),
            })
    return rows


def _window_network(block, n_active):
    """Within-window directed adjacency (dcor → percentile gate). None if too sparse."""
    idc = _idcor_window(block)
    max_i = np.maximum(idc, idc.T)[np.triu_indices(n_active, k=1)]
    if (max_i > 0).sum() < WF_K:
        return None
    adj = _build_adjacency(idc, float(np.percentile(max_i, WF_DEP_PCT)))
    return adj if (adj > 0).sum() >= WF_K else None


def _process_window(t_idx, date, rr, sem_sim, remove_factor):
    """All forecasts produced at one (forecast-date, factor-setting). Returns row dicts."""
    train = rr[t_idx - W_TRAIN:t_idx, :]                 # W × n raw returns
    active = np.nonzero(((~np.isnan(train)).sum(axis=0) >= WF_MIN_ACTIVE)
                        & ~np.isnan(rr[t_idx, :]))[0]     # point-in-time active gate
    if len(active) < 3 * WF_K:
        return []

    block = winsorize(train[:, active], WINSOR_LO, WINSOR_HI)
    if remove_factor:
        block = _residualize_pc1(block)

    adj = _window_network(block, len(active))
    if adj is None:
        return []

    labels = _cluster_window(adj, WF_K)
    means = _cluster_means(block, labels, WF_K)
    real = _realized(rr, t_idx, active, labels)
    rng = np.random.default_rng(RANDOM_SEED + t_idx)
    ctx = {"R": block, "rr": rr, "t_idx": t_idx, "active": active, "date": date,
           "remove_factor": remove_factor, "sem_active": sem_sim[np.ix_(active, active)]}
    rows = []
    for variant in ("unfiltered", "filtered", "random"):
        rows += _variant_rows(variant, adj, labels, means, real, rng, ctx)
    return rows


def _edge_forecast_with_lead(cma, cmb, realized):
    """Wrapper: predictor = cma at index (W - l*), i.e. the leading cluster's in-window
    return l* days before the forecast date (known at t)."""
    w = len(cma)
    best_l, best_d = 0, -1.0
    for l in range(1, MAX_LAG + 1):
        x, y = cma[:w - l], cmb[l:]
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 10:
            continue
        d = _abs_dcor(x[mask], y[mask])
        if d > best_d:
            best_d, best_l = d, l
    if best_l == 0 or np.isnan(realized):
        return None
    x, y = cma[:w - best_l], cmb[best_l:]
    mask = ~np.isnan(x) & ~np.isnan(y)
    lead_val = cma[w - best_l]                          # r̄^Ca at day t-l* (in-window, known)
    if mask.sum() < 10 or np.std(x[mask]) == 0 or np.isnan(lead_val):
        return None
    beta, alpha = np.polyfit(x[mask], y[mask], 1)
    return best_l, float(alpha + beta * lead_val)


def main():
    log.info("=" * 60)
    log.info("STEP 3.1–3.3: Walk-forward forecasting engine")
    log.info("=" * 60)

    logit_df = pd.read_parquet(LOGIT_PRICES_FILE)
    dates = list(logit_df.index)
    token_ids = list(logit_df.columns)
    logit = logit_df.to_numpy(dtype=float)
    # raw returns aligned to day index (row i = logit_i − logit_{i−1}); row 0 = NaN
    rr = np.full_like(logit, np.nan)
    rr[1:, :] = logit[1:, :] - logit[:-1, :]
    sem_sim = pd.read_parquet(SEM_SIM_FILE).reindex(index=token_ids, columns=token_ids).to_numpy()

    n_days = len(dates)
    forecast_idx = list(range(W_TRAIN, n_days, WF_STEP))
    tasks = [(t, f) for t in forecast_idx for f in WF_REMOVE_COMMON_FACTOR]
    log.info(f"{len(forecast_idx)} forecast dates × {len(WF_REMOVE_COMMON_FACTOR)} factor settings "
             f"= {len(tasks)} windows (W={W_TRAIN}, step={WF_STEP}, k={WF_K})")

    t0 = time.time()
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_process_window)(t, dates[t], rr, sem_sim, f) for t, f in tasks
    )
    rows = [r for window_rows in results for r in window_rows]
    log.info(f"Generated {len(rows):,} forecasts in {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    ensure_dirs(PHASE3_DIR)
    df.to_parquet(FORECASTS_FILE, index=False)
    log.info(f"Saved forecasts: {FORECASTS_FILE}")

    # Quick sanity: per-variant directional hit-rate (raw, pre-evaluation)
    if len(df):
        log.info("\n── SANITY (raw directional hit-rate, factor_removed=False) ──")
        sub = df[~df["factor_removed"]]
        for v in ("unfiltered", "filtered", "random"):
            d = sub[sub["variant"] == v]
            if len(d):
                hit = (np.sign(d["forecast"]) == np.sign(d["realized"])).mean()
                log.info(f"  {v:11s}: n={len(d):5d}  hit-rate={100*hit:.1f}%")
    return df


if __name__ == "__main__":
    main()
