"""
check_common_factor.py — common-factor (election-beta) robustness pre-check
===========================================================================
Tests whether the Phase 1/2 headline findings survive removing the dominant
common factor from returns. In an election cycle most markets co-move with one
latent signal ("who wins"); lead-lag between two markets that both track it (one
reacting faster) can look directional without genuine information transfer.

Procedure:
  1. Residualize: f_t = first principal component of returns (SVD on the
     zero-filled, column-centered return matrix). Each contract's returns are
     regressed on f over its active days; we keep the residuals.
  2. Re-run the lead-lag → adjacency → Hermitian-clustering → leadingness →
     three-way-ARI pipeline on the RESIDUALS, reusing the Phase 1/2 functions.
  3. Compare to the saved full-sample (original) results.

Decision: if the leader/follower hierarchy and the topic-orthogonality persist,
the structure is not merely common-factor reaction-speed → proceed to Phase 3.

Output: data/processed/phase1/common_factor_check.json
        data/processed/presentation/06_common_factor_check.png
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.linalg import eigh
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from config import (
    RETURNS_FILE, IDCOR_MATRIX_FILE, TAU_DEP_FILE, ADJACENCY_FILE, CLUSTERS_FILE,
    SEM_CLUSTERS_FILE, ELIGIBLE_MARKETS_FILE, THREEWAY_FILE,
    MAX_LAG, MIN_OVERLAP_DAYS, N_JOBS, PAIR_CHUNK_SIZE, RANDOM_SEED,
    DEP_FLOOR_PCT, PHASE1_DIR, PRESENTATION_DIR,
)
from utils import get_logger, ensure_dirs, save_json, load_json, mean_acf_block_length
from step1_1_leadlag import _directional_integral, _abs_dcor
from step1_2_significance import _coactive, _null_max_i_for_pair
from step1_6_liquidity import _build_adjacency, _node_leadingness
from step1_4_spectral import _normalized_hermitian, _top_positive, _embeddings, _cluster

log = get_logger("common_factor_check")

# Reduced bootstrap for the residualized τ_dep (the original used B=1000×1000).
RESID_B = 300
RESID_NULL_PAIRS = 500
K = 7                     # fixed to the finer structure used downstream


def residualize_pc1(returns_df):
    """Remove the first principal component (common factor) from each series."""
    R = returns_df.to_numpy(dtype=float)
    filled = np.nan_to_num(R, nan=0.0)
    centered = filled - filled.mean(axis=0, keepdims=True)
    U, s, _ = np.linalg.svd(centered, full_matrices=False)
    f = U[:, 0] * s[0]                                   # PC1 time series (n_days,)
    var_explained = float(s[0] ** 2 / (s ** 2).sum())

    resid = np.full_like(R, np.nan)
    design = np.column_stack([np.ones_like(f), f])
    for j in range(R.shape[1]):
        col = R[:, j]
        m = ~np.isnan(col)
        if m.sum() < 3:
            continue
        beta, *_ = np.linalg.lstsq(design[m], col[m], rcond=None)
        resid[m, j] = col[m] - design[m] @ beta
    return resid, var_explained


def _idcor_chunk(chunk, R):
    out = []
    for i, j in chunk:
        ri, rj = R[:, i], R[:, j]
        co = (~np.isnan(ri)) & (~np.isnan(rj))
        if co.sum() < MIN_OVERLAP_DAYS:
            continue
        out.append((i, j,
                    _directional_integral(ri, rj, MAX_LAG, _abs_dcor),
                    _directional_integral(rj, ri, MAX_LAG, _abs_dcor)))
    return out


def compute_idcor(R, pairs):
    n = R.shape[1]
    chunks = [pairs[k:k + PAIR_CHUNK_SIZE] for k in range(0, len(pairs), PAIR_CHUNK_SIZE)]
    results = Parallel(n_jobs=N_JOBS)(delayed(_idcor_chunk)(c, R) for c in chunks)
    idcor = np.zeros((n, n))
    for chunk_rows in results:
        for i, j, iij, iji in chunk_rows:
            idcor[i, j] = iij
            idcor[j, i] = iji
    return idcor


def resid_tau_dep(R, pairs, block_len, seed):
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(pairs), min(RESID_NULL_PAIRS, len(pairs)), replace=False)
    draws = []
    for idx, s in enumerate(sel):
        i, j = pairs[s]
        x, y = _coactive(R[:, i], R[:, j])
        if len(x) < MIN_OVERLAP_DAYS:
            continue
        draws.append(_null_max_i_for_pair(x, y, block_len, RESID_B, seed + idx))
    pool = np.concatenate(draws)
    return float(np.percentile(pool, DEP_FLOOR_PCT))


def hermitian_clusters(adj, k):
    a_sym, _, _ = _normalized_hermitian(adj)
    eigvals, eigvecs = eigh(a_sym)
    emb = _embeddings(eigvals, eigvecs, k, row_normalize=True)
    return _cluster(emb, k), eigvals


def main():
    log.info("=" * 60)
    log.info("COMMON-FACTOR ROBUSTNESS PRE-CHECK")
    log.info("=" * 60)

    returns_df = pd.read_parquet(RETURNS_FILE)
    token_ids = list(returns_df.columns)
    n = len(token_ids)

    # Residualize against PC1
    resid, var_exp = residualize_pc1(returns_df)
    log.info(f"PC1 explains {100*var_exp:.1f}% of return variance (the common factor)")

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # ── Residualized lead-lag pipeline ───────────────────────────────────
    t0 = time.time()
    log.info("Computing residualized I_dcor matrix...")
    idcor_r = compute_idcor(resid, pairs)
    block_len = mean_acf_block_length(pd.DataFrame(resid, columns=token_ids))
    log.info(f"Computing residualized τ_dep (B={RESID_B}, {RESID_NULL_PAIRS} pairs)...")
    tau_r = resid_tau_dep(resid, pairs, block_len, RANDOM_SEED)
    adj_r = _build_adjacency(idcor_r, tau_r)
    lead_r = _node_leadingness(adj_r)
    labels_r, eig_r = hermitian_clusters(adj_r, K)
    log.info(f"Residualized pipeline done in {time.time()-t0:.0f}s")

    # ── Saved originals ──────────────────────────────────────────────────
    idcor_o = pd.read_parquet(IDCOR_MATRIX_FILE).to_numpy()
    tau_o = load_json(TAU_DEP_FILE)["tau_dep"]
    adj_o = pd.read_parquet(ADJACENCY_FILE).to_numpy()
    lead_o = _node_leadingness(adj_o)
    cats = np.array([{r["token_id"]: r.get("category", "")
                      for r in load_json(ELIGIBLE_MARKETS_FILE)}.get(t, "") for t in token_ids])
    _, cat_codes = np.unique(cats, return_inverse=True)
    sem_map = load_json(SEM_CLUSTERS_FILE)["labels"]
    semantic = np.array([sem_map[t] for t in token_ids])
    stat_o = np.array([load_json(CLUSTERS_FILE)["labels"][t] for t in token_ids])

    # ── Comparison metrics ───────────────────────────────────────────────
    def frac_clear(idcor, tau):
        iu = np.triu_indices(n, k=1)
        maxI = np.maximum(idcor, idcor.T)[iu]
        return 100 * (maxI > tau).mean()

    lead_corr = float(np.corrcoef(lead_o, lead_r)[0, 1])
    tw_o = load_json(THREEWAY_FILE)["pairwise"]
    res = {
        "pc1_var_explained": var_exp,
        "original": {
            "tau_dep": tau_o,
            "pct_pairs_clear_floor": frac_clear(idcor_o, tau_o),
            "n_edges": int((adj_o > 0).sum()),
            "stat_vs_semantic_ARI": tw_o["statistical_vs_semantic"]["ARI"],
            "stat_vs_category_ARI": tw_o["statistical_vs_category"]["ARI"],
        },
        "residualized": {
            "tau_dep": tau_r,
            "pct_pairs_clear_floor": frac_clear(idcor_r, tau_r),
            "n_edges": int((adj_r > 0).sum()),
            "stat_vs_semantic_ARI": float(adjusted_rand_score(labels_r, semantic)),
            "stat_vs_category_ARI": float(adjusted_rand_score(labels_r, cat_codes)),
        },
        "leadingness_corr_orig_resid": lead_corr,
    }

    log.info("\n── ORIGINAL vs RESIDUALIZED ──")
    log.info(f"  PC1 (common factor) variance share: {100*var_exp:.1f}%")
    o, r = res["original"], res["residualized"]
    log.info(f"  % pairs clearing τ_dep:   {o['pct_pairs_clear_floor']:.1f}%  →  "
             f"{r['pct_pairs_clear_floor']:.1f}%   (5% = chance)")
    log.info(f"  directed edges:           {o['n_edges']}  →  {r['n_edges']}")
    log.info(f"  leadingness corr (orig vs resid):  {lead_corr:+.3f}")
    log.info(f"  three-way  stat↔semantic ARI: {o['stat_vs_semantic_ARI']:.3f} → {r['stat_vs_semantic_ARI']:.3f}")
    log.info(f"  three-way  stat↔category ARI: {o['stat_vs_category_ARI']:.3f} → {r['stat_vs_category_ARI']:.3f}")

    survives = (r["pct_pairs_clear_floor"] > 7.0 and lead_corr > 0.4)
    verdict = ("STRUCTURE SURVIVES — not merely common-factor reaction speed; proceed to Phase 3"
               if survives else
               "STRUCTURE WEAKENS substantially under factor removal — interpret leadership with caution")
    log.info(f"\nVERDICT: {verdict}")
    res["verdict"] = verdict

    ensure_dirs([PHASE1_DIR, PRESENTATION_DIR])
    save_json(res, os.path.join(PHASE1_DIR, "common_factor_check.json"))

    # ── Figure ───────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.scatter(lead_o, lead_r, s=16, alpha=0.5, color="#2b6cb0")
    lim = max(np.abs(lead_o).max(), np.abs(lead_r).max()) * 1.05
    ax1.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax1.axhline(0, color="gray", lw=0.5); ax1.axvline(0, color="gray", lw=0.5)
    ax1.set_xlabel("node leadingness — original")
    ax1.set_ylabel("node leadingness — residualized")
    ax1.set_title(f"(a) Leadingness persists after factor removal (r={lead_corr:+.2f})")

    labels = ["% pairs\nclear τ_dep", "stat↔sem\nARI", "stat↔cat\nARI"]
    ov = [o["pct_pairs_clear_floor"], o["stat_vs_semantic_ARI"] * 100, o["stat_vs_category_ARI"] * 100]
    rv = [r["pct_pairs_clear_floor"], r["stat_vs_semantic_ARI"] * 100, r["stat_vs_category_ARI"] * 100]
    x = np.arange(len(labels))
    ax2.bar(x - 0.2, ov, 0.4, label="original", color="#a0aec0")
    ax2.bar(x + 0.2, rv, 0.4, label="residualized", color="#dd6b20")
    ax2.axhline(5, color="red", ls=":", lw=1, label="5% chance (floor)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("value  (ARI ×100)")
    ax2.set_title(f"(b) Key metrics — PC1 explains {100*var_exp:.0f}% of variance")
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig_path = os.path.join(PRESENTATION_DIR, "06_common_factor_check.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {fig_path}")

    return res


if __name__ == "__main__":
    main()
