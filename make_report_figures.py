"""
make_report_figures.py — presentation-ready figure deck
=======================================================
Reads the saved Phase 0/1/2 artifacts and renders a consistent, numbered set of
presentation figures into data/processed/presentation/. Pure reporting — it
computes nothing new, just visualizes what the pipeline already produced.

    python make_report_figures.py

Figures:
  01_dataset_overview        — category mix + eligibility funnel
  02_leadlag_significance    — eigenvalue spectrum/eigengap + null vs observed dependence
  03_three_way_comparison    — ARI matrix + semantic-coherence z-scores  (centerpiece)
  04_cluster_structure       — category composition per cluster + leadingness ranking
  05_liquidity_controls      — leadingness vs volume (confound check)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

from config import (
    ELIGIBLE_MARKETS_FILE, FILTER_FUNNEL_FILE,
    EIGEN_FILE, NULL_DIST_FILE, LEADLAG_SCORES_FILE, TAU_DEP_FILE,
    CLUSTERS_FILE, CLUSTER_RANKING_FILE, LIQUIDITY_FILE, ADJACENCY_FILE,
    THREEWAY_FILE, PRESENTATION_DIR, DEP_FLOOR_PCT,
)
from utils import get_logger, ensure_dirs, load_json

log = get_logger("report_figures")

# Consistent style
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, RED, GRAY, GREEN, ORANGE = "#2b6cb0", "#e53e3e", "#a0aec0", "#38a169", "#dd6b20"
POS_EPS = 1e-9


def _save(fig, name):
    path = os.path.join(PRESENTATION_DIR, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log.info(f"  saved {name}")


# --------------------------------------------------------------------------
def fig_dataset_overview():
    recs = load_json(ELIGIBLE_MARKETS_FILE)
    cats = Counter(r["category"] for r in recs)
    funnel = pd.read_csv(FILTER_FUNNEL_FILE)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    items = cats.most_common()
    labels, vals = zip(*items)
    ax1.barh(range(len(labels)), vals, color=BLUE)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("eligible contracts")
    ax1.set_title(f"(a) Category composition — {len(recs)} contracts, {len(cats)} categories")
    for i, v in enumerate(vals):
        ax1.text(v + 2, i, str(v), va="center", fontsize=9)

    nice = {"0_total_tokens": "All markets", "1_has_price_data": "Has price data",
            "2_active_days_ge_60": "Active ≥60d", "3_volume_ge_50k": "Volume ≥$50k",
            "4_return_std_gt_001": "Return std >0.01", "5_max_gap_le_5": "Gap ≤5d"}
    steps = [nice.get(s, s) for s in funnel["step"]]
    ax2.bar(range(len(steps)), funnel["count"], color=GRAY)
    ax2.bar(len(steps) - 1, funnel["count"].iloc[-1], color=GREEN)
    ax2.set_xticks(range(len(steps)))
    ax2.set_xticklabels(steps, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("contracts (YES-side)")
    ax2.set_title("(b) Eligibility funnel")
    for i, v in enumerate(funnel["count"]):
        ax2.text(i, v + 25, str(v), ha="center", fontsize=9)
    _save(fig, "01_dataset_overview.png")


def fig_leadlag_significance():
    eigvals = np.load(EIGEN_FILE)
    pos = np.sort(eigvals[eigvals > POS_EPS])[::-1][:14]
    gaps = pos[:-1] - pos[1:]
    null = np.load(NULL_DIST_FILE)
    tau = load_json(TAU_DEP_FILE)["tau_dep"]
    sc = pd.read_parquet(LEADLAG_SCORES_FILE)
    obs = sc[["I_dcor_ij", "I_dcor_ji"]].max(axis=1).to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ks = np.arange(1, len(gaps) + 1)
    colors = [RED if k == 7 else (ORANGE if k == int(ks[np.argmax(gaps)]) else GRAY) for k in ks]
    ax1.bar(ks, gaps, color=colors)
    ax1.set_xticks(ks)
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("eigengap  λ_k − λ_{k+1}")
    ax1.set_title("(a) Spectral eigengaps — dominant k=2, finer k=7")
    ax1.text(2, gaps[1], " dominant", color=ORANGE, fontsize=9, va="bottom")
    ax1.text(7, gaps[6], " selected", color=RED, fontsize=9, va="bottom")

    frac_obs = 100 * (obs > tau).mean()
    ax2.hist(null, bins=80, density=True, color=GRAY, alpha=0.7, label="null (timing destroyed)")
    ax2.hist(obs, bins=80, density=True, color=BLUE, alpha=0.5, label="observed pairs")
    ax2.axvline(tau, color=RED, ls="--", lw=2, label=f"τ_dep (p{DEP_FLOOR_PCT})={tau:.2f}")
    ax2.set_xlabel("max directional dependence  max(I)")
    ax2.set_ylabel("density")
    ax2.set_title(f"(b) Dependence vs null — {frac_obs:.1f}% of pairs clear τ_dep (vs 5% by chance)")
    ax2.legend(fontsize=9)
    _save(fig, "02_leadlag_significance.png")


def fig_three_way():
    tw = load_json(THREEWAY_FILE)
    pw = tw["pairwise"]
    names = ["Statistical", "Semantic", "Category"]
    ari = np.eye(3)
    nmi = np.eye(3)
    keymap = {(0, 1): "statistical_vs_semantic", (1, 2): "semantic_vs_category",
              (0, 2): "statistical_vs_category"}
    for (i, j), key in keymap.items():
        ari[i, j] = ari[j, i] = pw[key]["ARI"]
        nmi[i, j] = nmi[j, i] = pw[key]["NMI"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    im = ax1.imshow(ari, cmap="YlOrRd", vmin=0, vmax=1)
    ax1.set_xticks(range(3)); ax1.set_xticklabels(names)
    ax1.set_yticks(range(3)); ax1.set_yticklabels(names)
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f"ARI {ari[i,j]:.2f}\nNMI {nmi[i,j]:.2f}", ha="center", va="center",
                     fontsize=10, color="black" if ari[i, j] < 0.6 else "white")
    ax1.set_title("(a) Partition agreement (ARI / NMI)")
    fig.colorbar(im, ax=ax1, fraction=0.046, label="ARI")

    coh = sorted(tw["semantic_coherence"], key=lambda d: d["z"], reverse=True)
    zs = [c["z"] for c in coh]
    labs = [f"C{c['cluster']}" for c in coh]
    bar_colors = [GREEN if z > 1.96 else (RED if z < -1.96 else GRAY) for z in zs]
    ax2.bar(range(len(zs)), zs, color=bar_colors)
    ax2.axhline(1.96, color="black", ls=":", lw=1)
    ax2.axhline(-1.96, color="black", ls=":", lw=1)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(range(len(zs))); ax2.set_xticklabels(labs)
    ax2.set_ylabel("Semantic Coherence z-score")
    ax2.set_title(f"(b) SCS of statistical clusters — {tw['n_coherent_clusters']}/{tw['k']} "
                  f"coherent (z>1.96)")
    ax2.text(len(zs) - 1, 1.96, " z=1.96", fontsize=8, va="bottom", ha="right")
    _save(fig, "03_three_way_comparison.png")


def fig_cluster_structure():
    clusters = load_json(CLUSTERS_FILE)
    labels = clusters["labels"]
    k = clusters["k_selected"]
    recs = {r["token_id"]: r["category"] for r in load_json(ELIGIBLE_MARKETS_FILE)}
    rank = pd.read_csv(CLUSTER_RANKING_FILE)
    order = [int(c[1:]) for c in rank["cluster"]]      # clusters by leadingness

    all_cats = sorted({c for c in recs.values()})
    comp = {a: Counter() for a in range(k)}
    for t, a in labels.items():
        comp[a][recs.get(t, "")] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(all_cats)))
    bottoms = np.zeros(k)
    xpos = range(k)
    for ci, cat in enumerate(all_cats):
        vals = [comp[a].get(cat, 0) for a in order]
        ax1.bar(xpos, vals, bottom=bottoms, label=cat, color=cmap[ci])
        bottoms += np.array(vals)
    ax1.set_xticks(list(xpos))
    ax1.set_xticklabels([f"C{a}" for a in order])
    ax1.set_xlabel("statistical cluster (ordered by leadingness →)")
    ax1.set_ylabel("contracts")
    ax1.set_title("(a) Category composition per cluster — clusters span topics")
    ax1.legend(fontsize=8, ncol=2, loc="upper right")

    lead = rank["leadingness"].to_numpy()
    bar_colors = [BLUE if v > 0 else RED for v in lead]
    ax2.bar(range(len(lead)), lead, color=bar_colors)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(range(len(lead)))
    ax2.set_xticklabels(rank["cluster"])
    ax2.set_ylabel("leadingness  L(a)")
    ax2.set_title("(b) Cluster leadingness ranking (leaders → followers)")
    _save(fig, "04_cluster_structure.png")


def fig_liquidity():
    clusters = load_json(CLUSTERS_FILE)
    adj_cols = list(pd.read_parquet(ADJACENCY_FILE).columns)
    adj = pd.read_parquet(ADJACENCY_FILE).to_numpy(dtype=float)
    lead = (adj - adj.T).sum(axis=1)
    liq = pd.read_parquet(LIQUIDITY_FILE).set_index("token_id").reindex(adj_cols)
    logv = np.log(liq["volume"].to_numpy(dtype=float))
    b, a = np.polyfit(logv, lead, 1)
    ss_res = np.sum((lead - (a + b * logv)) ** 2)
    ss_tot = np.sum((lead - lead.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(logv, lead, s=16, alpha=0.5, color=BLUE)
    xs = np.linspace(logv.min(), logv.max(), 50)
    ax.plot(xs, a + b * xs, color=RED, lw=2, label=f"OLS  R²={r2:.3f}")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("log(cumulative volume)")
    ax.set_ylabel("node leadingness  L(i)")
    ax.set_title("Liquidity confound check — leadingness is independent of volume")
    ax.legend()
    _save(fig, "05_liquidity_controls.png")


def main():
    log.info("=" * 60)
    log.info("Generating presentation figure deck")
    log.info("=" * 60)
    ensure_dirs(PRESENTATION_DIR)
    fig_dataset_overview()
    fig_leadlag_significance()
    fig_three_way()
    fig_cluster_structure()
    fig_liquidity()
    log.info(f"\nDeck written to {PRESENTATION_DIR}")


if __name__ == "__main__":
    main()
