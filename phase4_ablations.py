"""
phase4_ablations.py — robustness / ablation studies
===================================================
Establishes that the headline Phase-1/2 findings are robust to the main analyst
choices, not artifacts of them. Pure robustness reporting: it rebuilds pipeline
stages with the existing Phase-1 functions and scores each variant against the
dcor baseline via ARI of clusters + Spearman leadingness rank-correlation.

  A. Dependence metric — dcor vs Pearson vs Kendall (recompute integrals,
     matched-density adjacency, re-cluster).
  B. Clustering method — Hermitian random-walk vs naive symmetrization.
  Report-only — k-sweep (clusters.json), liquidity (Step 1.6), common factor
     (check_common_factor.py), pulled into one table.

    python phase4_ablations.py

Output: data/processed/phase4/ablation_summary.{csv,json}
        data/processed/presentation/08_ablations.png
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.stats import spearmanr
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from joblib import Parallel, delayed

from config import (
    RETURNS_FILE, IDCOR_MATRIX_FILE, ADJACENCY_FILE, CLUSTERS_FILE,
    ELIGIBLE_MARKETS_FILE, LIQUIDITY_CONTROLS_FILE, COMMON_FACTOR_CHECK_FILE,
    ABLATION_SUMMARY_CSV, ABLATION_SUMMARY_JSON, PHASE4_DIR, PRESENTATION_DIR,
    MAX_LAG, MIN_OVERLAP_DAYS, PAIR_CHUNK_SIZE, N_JOBS, RANDOM_SEED, WF_K,
)
from utils import get_logger, ensure_dirs, save_json, load_json
from step1_1_leadlag import _directional_integral, _abs_pearson, _abs_kendall
from step1_6_liquidity import _build_adjacency, _node_leadingness
from step1_4_spectral import _normalized_hermitian, _embeddings, _cluster

log = get_logger("phase4_ablations")
K = WF_K   # 7


# --------------------------------------------------------------------------
def _integral_matrix(R, pairs, corr_fn):
    """Directed integral matrix for a correlation primitive (reuses step1_1 logic)."""
    n = R.shape[1]

    def _chunk(chunk):
        out = []
        for i, j in chunk:
            ri, rj = R[:, i], R[:, j]
            co = (~np.isnan(ri)) & (~np.isnan(rj))
            if co.sum() < MIN_OVERLAP_DAYS:
                continue
            out.append((i, j,
                        _directional_integral(ri, rj, MAX_LAG, corr_fn),
                        _directional_integral(rj, ri, MAX_LAG, corr_fn)))
        return out

    chunks = [pairs[k:k + PAIR_CHUNK_SIZE] for k in range(0, len(pairs), PAIR_CHUNK_SIZE)]
    results = Parallel(n_jobs=N_JOBS)(delayed(_chunk)(c) for c in chunks)
    mat = np.zeros((n, n))
    for rows in results:
        for i, j, a, b in rows:
            mat[i, j] = a
            mat[j, i] = b
    return mat


def _matched_density_tau(idcor, n_target_edges):
    """Percentile threshold on max-I that yields ~n_target_edges directed edges."""
    n = idcor.shape[0]
    max_i = np.maximum(idcor, idcor.T)[np.triu_indices(n, k=1)]
    # each retained undirected pair → one directed edge; target the matching quantile
    frac_keep = n_target_edges / len(max_i)
    return float(np.quantile(max_i, 1.0 - frac_keep))


def _hermitian_clusters(adj, k):
    a_sym, _, _ = _normalized_hermitian(adj)
    eigvals, eigvecs = eigh(a_sym)
    emb = _embeddings(eigvals, eigvecs, k, row_normalize=True)
    return _cluster(emb, k)


def _symmetric_clusters(adj, k):
    """Naive symmetrization: cluster (A+Aᵀ)/2 with standard undirected spectral clustering."""
    a_sym = np.maximum((adj + adj.T) / 2.0, 0.0)
    np.fill_diagonal(a_sym, 0.0)
    sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                            assign_labels="kmeans", random_state=RANDOM_SEED)
    return sc.fit_predict(a_sym)


def _compare(name, labels, leadingness, base_labels, base_lead, n_edges, key_stat):
    ari = float(adjusted_rand_score(base_labels, labels))
    rho = float(spearmanr(base_lead, leadingness).correlation) if leadingness is not None else np.nan
    return {"ablation": name, "ari_to_baseline": ari, "leadingness_spearman": rho,
            "n_edges": int(n_edges), "key_stat": key_stat}


# --------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("PHASE 4: Ablation / robustness studies")
    log.info("=" * 60)

    returns_df = pd.read_parquet(RETURNS_FILE)
    token_ids = list(returns_df.columns)
    R = returns_df.to_numpy(dtype=float)
    n = len(token_ids)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Baseline (dcor)
    idcor = pd.read_parquet(IDCOR_MATRIX_FILE).to_numpy()
    adj_base = pd.read_parquet(ADJACENCY_FILE).to_numpy()
    base_edges = int((adj_base > 0).sum())
    base_labels = np.array([load_json(CLUSTERS_FILE)["labels"][t] for t in token_ids])
    base_lead = _node_leadingness(adj_base)
    log.info(f"Baseline: dcor, {base_edges} edges, k={K}")

    rows = []

    # ── Ablation A: dependence metric ────────────────────────────────────
    log.info("\n── A. Dependence metric (dcor vs Pearson vs Kendall) ──")
    for name, fn in [("Pearson", _abs_pearson), ("Kendall", _abs_kendall)]:
        t0 = time.time()
        mat = _integral_matrix(R, pairs, fn)
        tau = _matched_density_tau(mat, base_edges)
        adj = _build_adjacency(mat, tau)
        labels = _hermitian_clusters(adj, K)
        lead = _node_leadingness(adj)
        r = _compare(f"metric: {name}", labels, lead, base_labels, base_lead,
                     (adj > 0).sum(), f"vs dcor")
        rows.append(r)
        log.info(f"  {name:8s}: ARI_to_dcor={r['ari_to_baseline']:+.3f}  "
                 f"leadingness_ρ={r['leadingness_spearman']:+.3f}  "
                 f"edges={r['n_edges']}  ({time.time()-t0:.0f}s)")

    # ── Ablation B: clustering method ────────────────────────────────────
    log.info("\n── B. Clustering method (Hermitian vs naive symmetrization) ──")
    sym_labels = _symmetric_clusters(adj_base, K)
    ari_sym = float(adjusted_rand_score(base_labels, sym_labels))
    rows.append({"ablation": "method: symmetric (vs Hermitian)", "ari_to_baseline": ari_sym,
                 "leadingness_spearman": np.nan, "n_edges": base_edges,
                 "key_stat": "ARI(Hermitian, symmetric)"})
    log.info(f"  symmetric: ARI_to_Hermitian={ari_sym:+.3f}  "
             f"(low ⇒ direction matters; high ⇒ symmetrization suffices)")

    # ── Report-only rows ─────────────────────────────────────────────────
    log.info("\n── Report-only (already computed) ──")
    cats = np.array([{r["token_id"]: r.get("category", "")
                      for r in load_json(ELIGIBLE_MARKETS_FILE)}.get(t, "") for t in token_ids])
    _, cat_codes = np.unique(cats, return_inverse=True)
    lbk = load_json(CLUSTERS_FILE)["labels_by_k"]
    for kk in sorted(lbk, key=int):
        labs = np.array([lbk[kk][t] for t in token_ids])
        ari_cat = float(adjusted_rand_score(cat_codes, labs))
        ari_b = float(adjusted_rand_score(base_labels, labs))
        rows.append({"ablation": f"k={kk}", "ari_to_baseline": ari_b,
                     "leadingness_spearman": np.nan, "n_edges": base_edges,
                     "key_stat": f"ARI_vs_category={ari_cat:.3f}"})
        log.info(f"  k={kk:>2s}: ARI_vs_category={ari_cat:+.3f}  ARI_vs_k7={ari_b:+.3f}")

    cf = load_json(COMMON_FACTOR_CHECK_FILE)
    rows.append({"ablation": "common-factor removed", "ari_to_baseline": np.nan,
                 "leadingness_spearman": cf["leadingness_corr_orig_resid"],
                 "n_edges": cf["residualized"]["n_edges"],
                 "key_stat": f"PC1={100*cf['pc1_var_explained']:.1f}%var; struct survives"})
    log.info(f"  common-factor: PC1={100*cf['pc1_var_explained']:.1f}% var, "
             f"leadingness_ρ={cf['leadingness_corr_orig_resid']:+.3f} (survives)")

    liq = load_json(LIQUIDITY_CONTROLS_FILE)
    rows.append({"ablation": "liquidity (Step 1.6)", "ari_to_baseline": np.nan,
                 "leadingness_spearman": np.nan, "n_edges": base_edges,
                 "key_stat": f"leadingness~vol R2={liq['regression']['r2']:.3f}; persists in terciles"})
    log.info(f"  liquidity: R2(leadingness~log vol)={liq['regression']['r2']:.3f}")

    # ── Save ─────────────────────────────────────────────────────────────
    ensure_dirs([PHASE4_DIR, PRESENTATION_DIR])
    df = pd.DataFrame(rows)
    df.to_csv(ABLATION_SUMMARY_CSV, index=False)
    save_json({"baseline": "dcor / Hermitian / k=7", "rows": rows}, ABLATION_SUMMARY_JSON)
    log.info(f"\nSaved: {ABLATION_SUMMARY_CSV}")

    _figure(rows)
    return df


def _figure(rows):
    metric_rows = [r for r in rows if r["ablation"].startswith(("metric:", "method:"))]
    names = [r["ablation"].replace("metric: ", "").replace("method: ", "") for r in metric_rows]
    aris = [r["ari_to_baseline"] for r in metric_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#38a169" if a > 0.5 else "#dd6b20" for a in aris]
    ax.barh(range(len(names)), aris, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(1.0, color="gray", ls="--", lw=1, label="identical to baseline")
    ax.axvline(0.5, color="#a0aec0", ls=":", lw=1)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("ARI vs dcor/Hermitian baseline")
    ax.set_title("Phase 4 ablations — agreement of alternative methods with the baseline")
    for i, a in enumerate(aris):
        ax.text(a + 0.01, i, f"{a:.2f}", va="center", fontsize=9)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(PRESENTATION_DIR, "08_ablations.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
