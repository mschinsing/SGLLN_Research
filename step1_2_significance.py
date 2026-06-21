"""
Step 1.2 — Significance testing via stationary block bootstrap
==============================================================
Builds a single global dependence floor τ_dep under the null "no cross-series
lead–lag", using the stationary block bootstrap (Politis & Romano, 1994).

Procedure:
  1. Tune the expected block length from the data's lag-1 autocorrelation
     (utils.mean_acf_block_length) so resampling preserves realistic persistence.
  2. Sample NULL_SAMPLE_PAIRS random eligible pairs (overlap >= MIN_OVERLAP_DAYS).
  3. For each sampled pair, restrict to its co-active window, then for each of
     BOOTSTRAP_B replicates independently block-bootstrap each series' time index
     — this preserves within-series autocorrelation while destroying cross-series
     timing — and recompute max(I_ij, I_ji) using the SAME integrals as step 1.1.
  4. Pool all replicate statistics; τ_dep = percentile(pool, DEP_FLOOR_PCT).

The null is identical across pairs (independence of timing), so a pooled global
floor is valid and matches the proposal's singular "τ_dep".

Input:  data/processed/returns.parquet, data/processed/overlap_matrix.parquet
        data/processed/phase1/leadlag_scores.parquet   (for the gating preview)
Output: data/processed/phase1/null_distribution.npy
        data/processed/phase1/tau_dep.json
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import (
    RETURNS_FILE, OVERLAP_MATRIX_FILE, LEADLAG_SCORES_FILE,
    NULL_DIST_FILE, TAU_DEP_FILE, PHASE1_DIR,
    BOOTSTRAP_B, NULL_SAMPLE_PAIRS, DEP_FLOOR_PCT, MAX_LAG,
    MIN_OVERLAP_DAYS, RANDOM_SEED, N_JOBS,
)
from utils import (
    get_logger, ensure_dirs, save_json,
    stationary_block_bootstrap_indices, mean_acf_block_length,
)
# Reuse step 1.1's integral + dcor so the null statistic matches the observed one.
from step1_1_leadlag import _directional_integral, _abs_dcor

log = get_logger("step1.2_significance")


def _coactive(ri, rj):
    """Days where both series have a return; returns two equal-length clean arrays."""
    co = (~np.isnan(ri)) & (~np.isnan(rj))
    return ri[co], rj[co]


def _null_max_i_for_pair(x, y, block_len, n_reps, seed):
    """n_reps bootstrap draws of max(I_ij, I_ji) under independent block resampling."""
    rng = np.random.default_rng(seed)
    m = len(x)
    out = np.empty(n_reps, dtype=float)
    for b in range(n_reps):
        ix = stationary_block_bootstrap_indices(m, block_len, rng)
        iy = stationary_block_bootstrap_indices(m, block_len, rng)   # independent → kills timing
        xb, yb = x[ix], y[iy]
        i_ij = _directional_integral(xb, yb, MAX_LAG, _abs_dcor)
        i_ji = _directional_integral(yb, xb, MAX_LAG, _abs_dcor)
        out[b] = max(i_ij, i_ji)
    return out


def _null_chunk(chunk, returns, block_len, n_reps):
    """Pool null draws over a chunk of (i_idx, j_idx, seed) triples."""
    parts = []
    for i, j, seed in chunk:
        x, y = _coactive(returns[:, i], returns[:, j])
        if len(x) < MIN_OVERLAP_DAYS:
            continue
        parts.append(_null_max_i_for_pair(x, y, block_len, n_reps, seed))
    return np.concatenate(parts) if parts else np.empty(0)


def _chunked(seq, size):
    for k in range(0, len(seq), size):
        yield seq[k:k + size]


def main():
    log.info("=" * 60)
    log.info("STEP 1.2: Significance testing → global dependence floor τ_dep")
    log.info("=" * 60)

    returns_df = pd.read_parquet(RETURNS_FILE)
    overlap_df = pd.read_parquet(OVERLAP_MATRIX_FILE)
    R = returns_df.to_numpy(dtype=float)
    n = R.shape[1]

    # Block length from data persistence
    block_len = mean_acf_block_length(returns_df)
    log.info(f"Tuned expected block length: {block_len:.0f} day(s)")

    # Eligible pairs → sample NULL_SAMPLE_PAIRS of them (reproducibly)
    ov = overlap_df.to_numpy()
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if ov[i, j] >= MIN_OVERLAP_DAYS]
    rng = np.random.default_rng(RANDOM_SEED)
    n_sample = min(NULL_SAMPLE_PAIRS, len(pairs))
    sel = rng.choice(len(pairs), size=n_sample, replace=False)
    # attach a deterministic per-pair seed so results are chunk-order independent
    sampled = [(pairs[s][0], pairs[s][1], RANDOM_SEED + int(s)) for s in sel]
    log.info(f"Sampled {n_sample:,} pairs; {BOOTSTRAP_B} bootstrap reps each "
             f"(~{n_sample*BOOTSTRAP_B:,} pooled null draws)")

    # Chunk small enough to keep all workers busy (>> n_workers chunks)
    n_workers = os.cpu_count() or 4
    chunk_size = max(1, n_sample // (n_workers * 5))
    chunks = list(_chunked(sampled, chunk_size))
    log.info(f"Dispatching {len(chunks)} chunks of ~{chunk_size} pairs...")

    t0 = time.time()
    parts = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(_null_chunk)(chunk, R, block_len, BOOTSTRAP_B) for chunk in chunks
    )
    null = np.concatenate([p for p in parts if p.size])
    elapsed = time.time() - t0
    log.info(f"Built null of {null.size:,} draws in {elapsed:.1f}s")

    tau_dep = float(np.percentile(null, DEP_FLOOR_PCT))

    ensure_dirs(PHASE1_DIR)
    np.save(NULL_DIST_FILE, null)
    save_json({
        "tau_dep": tau_dep,
        "dep_floor_pct": DEP_FLOOR_PCT,
        "block_len": float(block_len),
        "n_pairs_sampled": n_sample,
        "bootstrap_B": BOOTSTRAP_B,
        "n_null_draws": int(null.size),
        "max_lag": MAX_LAG,
    }, TAU_DEP_FILE)
    log.info(f"Saved null distribution: {NULL_DIST_FILE}")
    log.info(f"Saved τ_dep: {TAU_DEP_FILE}")

    # Diagnostics: null shape + how many observed pairs clear the floor
    log.info("\n── SUMMARY ──")
    log.info(f"  Null max-I  mean/median: {null.mean():.4f} / {np.median(null):.4f}")
    log.info(f"  τ_dep (p{DEP_FLOOR_PCT}):          {tau_dep:.4f}")
    if os.path.exists(LEADLAG_SCORES_FILE):
        sc = pd.read_parquet(LEADLAG_SCORES_FILE)
        obs_max = sc[["I_dcor_ij", "I_dcor_ji"]].max(axis=1)
        n_pass = int((obs_max > tau_dep).sum())
        log.info(f"  Observed pairs > τ_dep:  {n_pass:,} / {len(sc):,} "
                 f"({100*n_pass/len(sc):.1f}%)  ← survive into adjacency (step 1.3)")

    return {"tau_dep": tau_dep, "block_len": block_len, "null": null}


if __name__ == "__main__":
    main()
