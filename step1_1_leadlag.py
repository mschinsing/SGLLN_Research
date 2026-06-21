"""
Step 1.1 — Pairwise lead–lag metric computation
================================================
For every eligible contract pair (i, j), compute directional dependence
integrals across lags l = 1..MAX_LAG and collapse them into a signed CCF-AUC
score that says who leads whom and how cleanly:

    I(i->j) = Σ_{l=1..L} |corr(r^i_{t-l}, r^j_t)|        (i leads j)
    I(j->i) = Σ_{l=1..L} |corr(r^j_{t-l}, r^i_t)|        (j leads i)
    S_auc   = sign(I_ij - I_ji) · max(I_ij, I_ji) / (I_ij + I_ji)

The primary correlation is distance correlation (dcor); Pearson and Kendall
CCF-AUC and a directional Granger p-value are computed as baselines.

Input:  data/processed/returns.parquet      (158 days × 354 contracts)
        data/processed/overlap_matrix.parquet
Output: data/processed/phase1/leadlag_scores.parquet  (long table, one row/pair)
        data/processed/phase1/idcor_matrix.parquet     (dense 354×354 I_dcor, for 1.2/1.3)

Pairs are processed in chunks across CPU cores with joblib; chunking (rather
than one job per pair) keeps per-task dispatch overhead negligible.
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import dcor
from scipy.stats import pearsonr, kendalltau
from joblib import Parallel, delayed

from config import (
    RETURNS_FILE, OVERLAP_MATRIX_FILE,
    LEADLAG_SCORES_FILE, IDCOR_MATRIX_FILE, PHASE1_DIR,
    MAX_LAG, MIN_OVERLAP_DAYS, PAIR_CHUNK_SIZE, N_JOBS,
)
from utils import get_logger, ensure_dirs

log = get_logger("step1.1_leadlag")


# --------------------------------------------------------------------------
# Correlation primitives
# --------------------------------------------------------------------------
def _abs_pearson(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = pearsonr(x, y)[0]
    return abs(r) if np.isfinite(r) else 0.0


def _abs_kendall(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = kendalltau(x, y)[0]
    return abs(t) if np.isfinite(t) else 0.0


def _abs_dcor(x, y):
    # distance correlation is already in [0, 1]; guard degenerate input.
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    try:
        return float(dcor.distance_correlation(x, y))
    except Exception:
        return 0.0


def _directional_integral(lead, follow, L, corr_fn):
    """I = Σ_{l=1..L} |corr(lead_{t-l}, follow_t)| over co-present days at each lag.

    `lead` and `follow` are equal-length arrays already restricted to the pair's
    co-active window (may still contain NaN from non-overlapping days)."""
    total = 0.0
    n = len(lead)
    for l in range(1, L + 1):
        if n - l < 3:
            break
        x = lead[:n - l]      # lead_{t-l}
        y = follow[l:]        # follow_t
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 3:
            continue
        total += corr_fn(x[mask], y[mask])
    return total


def _ccf_auc(i_fwd, i_rev):
    """sign(i_fwd - i_rev) · max / (i_fwd + i_rev); 0 when there's no dependence."""
    denom = i_fwd + i_rev
    if denom <= 0:
        return 0.0
    return float(np.sign(i_fwd - i_rev) * max(i_fwd, i_rev) / denom)


def _granger_dir(lead, follow, L):
    """Directional Granger: does `lead` Granger-cause `follow`?
    Returns the min p-value over lags 1..L on the NaN-dropped co-active series,
    or NaN if the test can't be run (short/singular series)."""
    from statsmodels.tsa.stattools import grangercausalitytests
    mask = ~np.isnan(lead) & ~np.isnan(follow)
    x, y = lead[mask], follow[mask]
    if len(y) < 5 * L or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    # grangercausalitytests([:, [y, x]]) tests whether x causes y.
    data = np.column_stack([y, x])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(data, maxlag=L, verbose=False)
        pvals = [res[l][0]["ssr_ftest"][1] for l in range(1, L + 1)]
        pvals = [p for p in pvals if np.isfinite(p)]
        return float(min(pvals)) if pvals else np.nan
    except Exception:
        return np.nan


# --------------------------------------------------------------------------
# Per-pair scoring
# --------------------------------------------------------------------------
def _score_pair(i, j, R, token_ids):
    """Compute all directional metrics for contract pair (i, j).
    Returns a result dict, or None if the pair fails the runtime overlap check."""
    ri = R[:, i]
    rj = R[:, j]

    # Defensive co-active overlap check — don't trust the Phase 0 matrix blindly.
    co = (~np.isnan(ri)) & (~np.isnan(rj))
    if co.sum() < MIN_OVERLAP_DAYS:
        return None

    # dcor integrals (primary)
    i_dcor_ij = _directional_integral(ri, rj, MAX_LAG, _abs_dcor)
    i_dcor_ji = _directional_integral(rj, ri, MAX_LAG, _abs_dcor)

    # baseline integrals
    i_p_ij = _directional_integral(ri, rj, MAX_LAG, _abs_pearson)
    i_p_ji = _directional_integral(rj, ri, MAX_LAG, _abs_pearson)
    i_k_ij = _directional_integral(ri, rj, MAX_LAG, _abs_kendall)
    i_k_ji = _directional_integral(rj, ri, MAX_LAG, _abs_kendall)

    return {
        "i": token_ids[i],
        "j": token_ids[j],
        "i_idx": i,
        "j_idx": j,
        "I_dcor_ij": i_dcor_ij,
        "I_dcor_ji": i_dcor_ji,
        "S_auc": _ccf_auc(i_dcor_ij, i_dcor_ji),
        "S_pearson": _ccf_auc(i_p_ij, i_p_ji),
        "S_kendall": _ccf_auc(i_k_ij, i_k_ji),
        "granger_ij": _granger_dir(ri, rj, MAX_LAG),
        "granger_ji": _granger_dir(rj, ri, MAX_LAG),
    }


def _score_chunk(chunk, R, token_ids):
    """Score a list of (i, j) pairs; returns (rows, n_skipped)."""
    rows, skipped = [], 0
    for i, j in chunk:
        res = _score_pair(i, j, R, token_ids)
        if res is None:
            skipped += 1
        else:
            rows.append(res)
    return rows, skipped


def _chunked(seq, size):
    for k in range(0, len(seq), size):
        yield seq[k:k + size]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("STEP 1.1: Pairwise lead–lag metric computation")
    log.info("=" * 60)

    returns_df = pd.read_parquet(RETURNS_FILE)
    overlap_df = pd.read_parquet(OVERLAP_MATRIX_FILE)
    token_ids = list(returns_df.columns)
    R = returns_df.to_numpy(dtype=float)            # (n_days, n_tokens)
    n = len(token_ids)
    log.info(f"Returns matrix: {R.shape} ({n} contracts)")

    # Enumerate unordered pairs that clear the overlap floor (matrix-level gate).
    ov = overlap_df.to_numpy()
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if ov[i, j] >= MIN_OVERLAP_DAYS]
    log.info(f"Eligible pairs (overlap >= {MIN_OVERLAP_DAYS}d): {len(pairs):,}")

    chunks = list(_chunked(pairs, PAIR_CHUNK_SIZE))
    log.info(f"Dispatching {len(chunks)} chunks of ~{PAIR_CHUNK_SIZE} pairs "
             f"across {N_JOBS} workers...")

    t0 = time.time()
    results = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(_score_chunk)(chunk, R, token_ids) for chunk in chunks
    )
    elapsed = time.time() - t0

    rows = [r for chunk_rows, _ in results for r in chunk_rows]
    skipped = sum(s for _, s in results)
    log.info(f"Scored {len(rows):,} pairs in {elapsed:.1f}s "
             f"({skipped} skipped by runtime overlap check)")

    df = pd.DataFrame(rows)

    # Dense I_dcor matrix (directed): entry [i, j] = I(i->j). For 1.2 / 1.3.
    idcor = np.zeros((n, n), dtype=float)
    for r in rows:
        idcor[r["i_idx"], r["j_idx"]] = r["I_dcor_ij"]
        idcor[r["j_idx"], r["i_idx"]] = r["I_dcor_ji"]

    ensure_dirs(PHASE1_DIR)
    df.drop(columns=["i_idx", "j_idx"]).to_parquet(LEADLAG_SCORES_FILE, index=False)
    pd.DataFrame(idcor, index=token_ids, columns=token_ids).to_parquet(IDCOR_MATRIX_FILE)
    log.info(f"Saved scores:      {LEADLAG_SCORES_FILE}")
    log.info(f"Saved I_dcor matrix: {IDCOR_MATRIX_FILE}")

    # Summary stats
    log.info("\n── SUMMARY ──")
    log.info(f"  Pairs scored:     {len(df):,}")
    log.info(f"  S_auc  mean/std:  {df['S_auc'].mean():+.4f} / {df['S_auc'].std():.4f}")
    log.info(f"  max(I_dcor) median: {df[['I_dcor_ij','I_dcor_ji']].max(axis=1).median():.4f}")
    log.info(f"  Granger NaN rate: {df['granger_ij'].isna().mean():.1%}")

    return df


if __name__ == "__main__":
    main()
