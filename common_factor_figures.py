"""
common_factor_figures.py — promote the common-factor robustness to a headline
=============================================================================
Concern #2 (election-beta confound) was the biggest threat to the central claim.
We already showed structure survives PC1 removal (leadingness ρ=0.75); this makes
it prominent with before/after visuals and an edge-survival statistic.

Rebuilds the residualized adjacency cheaply: recompute the residualized dcor
integral matrix (~17s) and REUSE the saved residualized τ_dep (no re-bootstrap).

  python common_factor_figures.py

Output: data/processed/phase1/common_factor_resid_adjacency.parquet
        data/processed/phase1/common_factor_strength.json
        data/processed/presentation/11_common_factor_strength.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from config import (
    RETURNS_FILE, ADJACENCY_FILE, CLUSTERS_FILE, COMMON_FACTOR_CHECK_FILE,
    COMMON_FACTOR_RESID_ADJ_FILE, COMMON_FACTOR_STRENGTH_FILE,
    PHASE1_DIR, PRESENTATION_DIR,
)
from utils import get_logger, ensure_dirs, save_json, load_json
from check_common_factor import residualize_pc1, compute_idcor
from step1_6_liquidity import _build_adjacency, _node_leadingness

log = get_logger("common_factor_figures")


def _cluster_leadingness(adj, labels, k):
    """Leadingness of each cluster under a given adjacency (fixed partition)."""
    net = adj - adj.T
    return np.array([net[np.ix_(np.nonzero(labels == a)[0], np.arange(adj.shape[0]))].sum()
                     / max((labels == a).sum(), 1) for a in range(k)])


def main():
    log.info("=" * 60)
    log.info("Common-factor robustness — before/after + edge survival")
    log.info("=" * 60)

    returns_df = pd.read_parquet(RETURNS_FILE)
    token_ids = list(returns_df.columns)
    n = len(token_ids)

    # Residualized adjacency (reuse saved τ_dep — no re-bootstrap)
    resid, var_exp = residualize_pc1(returns_df)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    log.info(f"Recomputing residualized I_dcor (PC1={100*var_exp:.1f}% var)...")
    idcor_r = compute_idcor(resid, pairs)
    tau_r = float(load_json(COMMON_FACTOR_CHECK_FILE)["residualized"]["tau_dep"])
    adj_r = _build_adjacency(idcor_r, tau_r)

    adj_o = pd.read_parquet(ADJACENCY_FILE).to_numpy(dtype=float)
    ensure_dirs([PHASE1_DIR, PRESENTATION_DIR])
    pd.DataFrame(adj_r, index=token_ids, columns=token_ids).to_parquet(COMMON_FACTOR_RESID_ADJ_FILE)

    # ── Edge survival ────────────────────────────────────────────────────
    eo = adj_o > 0
    er = adj_r > 0
    n_o, n_r = int(eo.sum()), int(er.sum())
    survived = int((eo & er).sum())                 # same node-pair AND same direction
    survival_frac = survived / n_o if n_o else 0.0
    log.info(f"\n── Edge survival ──")
    log.info(f"  original edges: {n_o}  | residualized: {n_r}  | survived (same direction): "
             f"{survived}  ({100*survival_frac:.1f}%)")

    # ── Cluster-ranking stability (fixed original k=7 partition) ─────────
    clusters = load_json(CLUSTERS_FILE)
    k = clusters["k_selected"]
    labels = np.array([clusters["labels"][t] for t in token_ids])
    lead_o = _cluster_leadingness(adj_o, labels, k)
    lead_r = _cluster_leadingness(adj_r, labels, k)
    rho = float(spearmanr(lead_o, lead_r).correlation)
    log.info(f"  cluster-leadingness ranking Spearman (orig vs resid): {rho:+.3f}")

    save_json({"pc1_var_explained": var_exp, "n_edges_original": n_o,
               "n_edges_residualized": n_r, "edges_survived": survived,
               "edge_survival_frac": survival_frac,
               "cluster_ranking_spearman": rho,
               "cluster_leadingness_original": lead_o.tolist(),
               "cluster_leadingness_residualized": lead_r.tolist()},
              COMMON_FACTOR_STRENGTH_FILE)

    _figure(adj_o, adj_r, labels, lead_o, lead_r, n_o, n_r, survived, survival_frac, rho, var_exp)
    log.info(f"\nSaved: {COMMON_FACTOR_STRENGTH_FILE}")
    return {"edge_survival_frac": survival_frac, "ranking_spearman": rho}


def _figure(adj_o, adj_r, labels, lead_o, lead_r, n_o, n_r, survived, frac, rho, var_exp):
    order = np.argsort(labels, kind="stable")       # group nodes by cluster for block view
    no = np.abs(adj_o - adj_o.T)[np.ix_(order, order)]
    nr = np.abs(adj_r - adj_r.T)[np.ix_(order, order)]
    vmax = np.percentile(no[no > 0], 99) if (no > 0).any() else 1.0

    fig = plt.figure(figsize=(14, 4.6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    for ax, mat, title in [(ax1, no, "original"), (ax2, nr, "residualized (PC1 removed)")]:
        ax.imshow(mat, cmap="magma", vmax=vmax, aspect="auto")
        ax.set_title(f"|net flow| — {title}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.34, 0.02, f"PC1 = {100*var_exp:.1f}% of variance · nodes ordered by cluster",
             ha="center", fontsize=8, color="gray")

    # Panel 3: edge survival + ranking
    ax3.barh([2], [n_o], color="#a0aec0", label="original")
    ax3.barh([1], [survived], color="#38a169", label="survived (same dir)")
    ax3.barh([0], [n_r], color="#dd6b20", label="residualized")
    ax3.set_yticks([2, 1, 0]); ax3.set_yticklabels(["original", "survived", "residualized"])
    ax3.set_xlabel("directed edges")
    ax3.set_title(f"Edge survival {100*frac:.0f}%  ·  cluster-rank ρ={rho:+.2f}", fontsize=10)
    for y, v in [(2, n_o), (1, survived), (0, n_r)]:
        ax3.text(v, y, f" {v}", va="center", fontsize=9)
    fig.suptitle("Common-factor robustness — structure survives election-beta removal",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(PRESENTATION_DIR, "11_common_factor_strength.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
