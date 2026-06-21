"""
Step 1.6 — Liquidity controls (confound check)
==============================================
Tests whether the discovered lead–lag structure is just a liquidity artifact
(high-volume markets updating first). Three checks:

  1. Volume-stratified replication: split contracts into volume terciles and
     rebuild the directed network *within* each bucket (reusing the dense I_dcor
     matrix + global τ_dep). If directional structure persists among
     similar-volume markets, it isn't merely a volume gradient.
  2. Leadingness-vs-volume regression: OLS of node leadingness L(i) on
     log(volume_i). A high R² would mean the ranking is mostly liquidity.
  3. Liquidity-label placebo: shuffle volume labels, re-fit the regression
     PERM_B times → null R² distribution; confirms the observed R² (if any)
     is non-trivial / collapses under shuffling.

`volume` is Gamma cumulative market volume (LIQUIDITY_FILE) — a liquidity proxy.

Input:  data/processed/liquidity_metadata.parquet
        data/processed/phase1/idcor_matrix.parquet
        data/processed/phase1/adjacency_matrix.parquet
        data/processed/phase1/tau_dep.json
Output: data/processed/phase1/liquidity_controls.json
        data/processed/phase1/figures/leadingness_vs_volume.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    LIQUIDITY_FILE, IDCOR_MATRIX_FILE, ADJACENCY_FILE, TAU_DEP_FILE,
    LIQUIDITY_CONTROLS_FILE, FIGURES_DIR, PHASE1_DIR,
    N_VOLUME_TERCILES, PERM_B, RANDOM_SEED,
)
from utils import get_logger, ensure_dirs, save_json, load_json

log = get_logger("step1.6_liquidity")


def _build_adjacency(idcor, tau_dep):
    """Same dependence-gated directed adjacency as step 1.3, on a (sub)matrix."""
    i_t = idcor.T
    m = np.maximum(idcor, i_t)
    denom = idcor + i_t
    ratio = np.divide(m, denom, out=np.zeros_like(m), where=denom > 0)
    s = np.sign(idcor - i_t) * ratio
    adj = np.maximum(s, 0.0) * m
    adj[m <= tau_dep] = 0.0
    np.fill_diagonal(adj, 0.0)
    return adj


def _node_leadingness(adj):
    """L(i) = Σ_j (A_ij − A_ji) = row sum of the net-flow matrix."""
    return (adj - adj.T).sum(axis=1)


def main():
    log.info("=" * 60)
    log.info("STEP 1.6: Liquidity controls (confound check)")
    log.info("=" * 60)

    idcor_df = pd.read_parquet(IDCOR_MATRIX_FILE)
    token_ids = list(idcor_df.columns)
    idcor = idcor_df.to_numpy(dtype=float)
    adj = pd.read_parquet(ADJACENCY_FILE).to_numpy(dtype=float)
    tau_dep = float(load_json(TAU_DEP_FILE)["tau_dep"])

    liq = pd.read_parquet(LIQUIDITY_FILE).set_index("token_id")
    volume = liq.reindex(token_ids)["volume"].to_numpy(dtype=float)
    log_vol = np.log(volume)
    leadingness = _node_leadingness(adj)
    n = len(token_ids)
    log.info(f"{n} nodes; τ_dep={tau_dep:.4f}; volume range "
             f"${volume.min():,.0f}–${volume.max():,.0f}")

    results = {"tau_dep": tau_dep, "n_nodes": n}

    # ── 1. Volume-stratified replication ─────────────────────────────────
    order = np.argsort(volume)
    tercile_of = np.empty(n, dtype=int)
    for t, idx in enumerate(np.array_split(order, N_VOLUME_TERCILES)):
        tercile_of[idx] = t
    full_density = (adj > 0).sum() / (n * (n - 1))
    log.info("\n── 1. Volume-stratified replication ──")
    log.info(f"  Full network: density={100*full_density:.2f}%")
    terciles = []
    for t in range(N_VOLUME_TERCILES):
        members = np.nonzero(tercile_of == t)[0]
        sub = _build_adjacency(idcor[np.ix_(members, members)], tau_dep)
        nt = len(members)
        edges = int((sub > 0).sum())
        density = edges / (nt * (nt - 1)) if nt > 1 else 0.0
        lead_std = float(_node_leadingness(sub).std())
        terciles.append({"tercile": t, "n": nt, "edges": edges,
                         "density": density, "leadingness_std": lead_std,
                         "vol_min": float(volume[members].min()),
                         "vol_max": float(volume[members].max())})
        log.info(f"  Tercile {t} (n={nt}, ${volume[members].min():,.0f}–"
                 f"${volume[members].max():,.0f}): density={100*density:.2f}%, "
                 f"edges={edges}, L-std={lead_std:.3f}")
    results["volume_stratified"] = terciles
    results["full_density"] = full_density

    # ── 2. Leadingness vs log(volume) regression ─────────────────────────
    x = sm.add_constant(log_vol)
    model = sm.OLS(leadingness, x).fit()
    obs_r2 = float(model.rsquared)
    slope = float(model.params[1])
    slope_p = float(model.pvalues[1])
    log.info("\n── 2. Leadingness ~ log(volume) OLS ──")
    log.info(f"  R²={obs_r2:.4f}  slope={slope:+.4f}  p={slope_p:.3g}")
    log.info("  (high R² ⇒ leadingness is largely a liquidity artifact)")
    results["regression"] = {"r2": obs_r2, "slope": slope, "slope_pvalue": slope_p}

    # ── 3. Liquidity-label placebo ───────────────────────────────────────
    rng = np.random.default_rng(RANDOM_SEED)
    null_r2 = np.empty(PERM_B)
    for b in range(PERM_B):
        xb = sm.add_constant(rng.permutation(log_vol))
        null_r2[b] = sm.OLS(leadingness, xb).fit().rsquared
    placebo_p = float((np.sum(null_r2 >= obs_r2) + 1) / (PERM_B + 1))
    log.info("\n── 3. Liquidity-label placebo ──")
    log.info(f"  Null R² mean={null_r2.mean():.4f} p95={np.percentile(null_r2,95):.4f}; "
             f"observed={obs_r2:.4f}; placebo p={placebo_p:.3g}")
    results["placebo"] = {"null_r2_mean": float(null_r2.mean()),
                          "null_r2_p95": float(np.percentile(null_r2, 95)),
                          "observed_r2": obs_r2, "placebo_pvalue": placebo_p}

    # ── Figure: leadingness vs log(volume) ───────────────────────────────
    ensure_dirs([PHASE1_DIR, FIGURES_DIR])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(log_vol, leadingness, s=14, alpha=0.5)
    xs = np.linspace(log_vol.min(), log_vol.max(), 50)
    ax.plot(xs, model.params[0] + slope * xs, "r-",
            label=f"OLS  R²={obs_r2:.3f}")
    ax.set_xlabel("log(cumulative volume)")
    ax.set_ylabel("node leadingness  L(i)")
    ax.set_title("Leadingness vs liquidity (confound check)")
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "leadingness_vs_volume.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    save_json(results, LIQUIDITY_CONTROLS_FILE)
    log.info(f"\nSaved liquidity controls: {LIQUIDITY_CONTROLS_FILE}")
    log.info(f"Saved figure: {fig_path}")

    # Verdict
    verdict = ("structure largely survives liquidity controls"
               if obs_r2 < 0.25 else "leadingness substantially tracks volume — interpret with caution")
    log.info(f"\nVERDICT: R²={obs_r2:.3f} → {verdict}")

    return results


if __name__ == "__main__":
    main()
