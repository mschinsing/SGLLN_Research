"""
Step 3.4 — Forecast evaluation
==============================
Evaluates the walk-forward forecasts across the three signal variants
(unfiltered / filtered / random) and both factor settings.

Metrics per (factor_removed, variant):
  * Directional accuracy (% correct sign) with bootstrap CI; one-sided binomial
    test vs 50%.
  * MSE with bootstrap CI.
  * Illustrative Sharpe — strategy return = sign(forecast)·realized, per-trade
    mean/std. GROSS only: no real bid-ask spreads exist for these markets, so
    this is a sanity number, not a tradeable claim.
  * Difference unfiltered−random and filtered−random (bootstrap CI on the gap):
    does real structure beat the permuted-label baseline?
  * FDR control (Benjamini–Hochberg) on per-edge directional claims.

Input:  data/processed/phase3/forecasts.parquet
Output: data/processed/phase3/forecast_metrics.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from config import FORECASTS_FILE, WF_METRICS_FILE, PHASE3_DIR, N_BOOTSTRAP_CI, RANDOM_SEED
from utils import get_logger, ensure_dirs, save_json

log = get_logger("step3.2_evaluate")
_RNG = np.random.default_rng(RANDOM_SEED)

# Cost model: strategy P&L is in log-odds return units; a round-trip spread of
# `s` price-units costs s/(p(1-p)) in log-odds units. P_REF=0.5 maximizes p(1-p),
# giving the SMALLEST log-odds cost → a GENEROUS (upper-bound) break-even spread.
COST_GRID_CENTS = [0.0, 0.5, 1.0, 2.0, 5.0]   # round-trip spread (cents)
P_REF = 0.5


def _cents_to_logodds(cents):
    return (cents / 100.0) / (P_REF * (1.0 - P_REF))


def _cost_analysis(d):
    """Break-even round-trip spread and net-Sharpe sensitivity for one variant."""
    strat = (np.sign(d["forecast"]) * d["realized"]).to_numpy()
    g = float(strat.mean())                       # gross mean per-trade return (log-odds)
    sens = [{"spread_cents": c,
             "net_mean": float((strat - _cents_to_logodds(c)).mean()),
             "net_sharpe": _sharpe(strat - _cents_to_logodds(c))}
            for c in COST_GRID_CENTS]
    return {
        "gross_mean_return": g,
        "gross_mean_ci": _boot_ci(strat, np.mean),
        # cost (in cents) that zeroes mean net P&L; <=0 means no positive cost is survivable
        "break_even_spread_cents": float(g * P_REF * (1.0 - P_REF) * 100.0),
        "cost_sensitivity": sens,
    }


def _boot_ci(values, stat_fn, b=N_BOOTSTRAP_CI):
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n < 5:
        return (np.nan, np.nan)
    draws = [stat_fn(vals[_RNG.integers(0, n, n)]) for _ in range(b)]
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def _sharpe(strat):
    return float(strat.mean() / strat.std()) if strat.std() > 0 else 0.0


def _metrics(d):
    f, r = d["forecast"].to_numpy(), d["realized"].to_numpy()
    hit = (np.sign(f) == np.sign(r)).astype(float)
    sq = (f - r) ** 2
    strat = np.sign(f) * r
    n = len(d)
    acc = float(hit.mean())
    return {
        "n": n,
        "dir_accuracy": acc,
        "dir_acc_ci": _boot_ci(hit, np.mean),
        "binom_p_gt_50": float(binomtest(int(hit.sum()), n, 0.5, alternative="greater").pvalue),
        "mse": float(sq.mean()),
        "mse_ci": _boot_ci(sq, np.mean),
        "sharpe_gross_illustrative": _sharpe(strat),
        "sharpe_ci": _boot_ci(strat, _sharpe),
        "cost": _cost_analysis(d),
    }


def _diff_vs_random(d_var, d_rand):
    """Bootstrap CI on (variant dir-acc − random dir-acc)."""
    hv = (np.sign(d_var["forecast"]) == np.sign(d_var["realized"])).to_numpy(float)
    hr = (np.sign(d_rand["forecast"]) == np.sign(d_rand["realized"])).to_numpy(float)
    if len(hv) < 5 or len(hr) < 5:
        return {"diff": np.nan, "ci": (np.nan, np.nan)}
    draws = [hv[_RNG.integers(0, len(hv), len(hv))].mean()
             - hr[_RNG.integers(0, len(hr), len(hr))].mean() for _ in range(N_BOOTSTRAP_CI)]
    return {"diff": float(hv.mean() - hr.mean()),
            "ci": (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))}


def _fdr_per_edge(d, min_n=10):
    """BH-FDR over per-edge (lead→lag) one-sided directional claims."""
    pvals, edges = [], []
    for (a, b), g in d.groupby(["lead_cluster", "lag_cluster"]):
        if len(g) < min_n:
            continue
        hit = (np.sign(g["forecast"]) == np.sign(g["realized"])).sum()
        pvals.append(binomtest(int(hit), len(g), 0.5, alternative="greater").pvalue)
        edges.append((int(a), int(b), len(g)))
    if not pvals:
        return {"n_edges_tested": 0, "n_significant_fdr": 0}
    reject, _, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    return {"n_edges_tested": len(pvals), "n_significant_fdr": int(reject.sum())}


def _evaluate_factor(sub, tag):
    """Metrics for all three variants at one factor setting; logs a table."""
    out = {}
    d_rand = sub[sub["variant"] == "random"]
    log.info(f"\n── {tag.upper()} ──")
    log.info(f"  {'variant':11s} {'n':>5s} {'dir_acc':>8s} {'95% CI':>16s} "
             f"{'binom_p':>8s} {'vs_random':>12s}")
    for variant in ("unfiltered", "filtered", "random"):
        d = sub[sub["variant"] == variant]
        if d.empty:
            continue
        m = _metrics(d)
        if variant != "random":
            m["vs_random"] = _diff_vs_random(d, d_rand)
            m["fdr"] = _fdr_per_edge(d)
        out[variant] = m
        ci = m["dir_acc_ci"]
        vr = f"{m['vs_random']['diff']:+.3f}" if variant != "random" else "—"
        log.info(f"  {variant:11s} {m['n']:5d} {100*m['dir_accuracy']:7.1f}% "
                 f"[{100*ci[0]:5.1f},{100*ci[1]:5.1f}] {m['binom_p_gt_50']:8.3f} {vr:>12s}")
    for variant in ("unfiltered", "filtered"):
        if variant in out:
            fdr = out[variant]["fdr"]
            log.info(f"    {variant} FDR: {fdr['n_significant_fdr']}/{fdr['n_edges_tested']} "
                     f"edges significant (BH 0.05)")
    # Break-even round-trip spread + net Sharpe at a 1¢ spread
    log.info("  break-even spread (¢, generous upper bound) / net-Sharpe @1¢:")
    for variant in ("unfiltered", "filtered", "random"):
        if variant in out:
            c = out[variant]["cost"]
            s1 = next(s["net_sharpe"] for s in c["cost_sensitivity"]
                      if abs(s["spread_cents"] - 1.0) < 1e-9)
            log.info(f"    {variant:11s}: break-even={c['break_even_spread_cents']:+.3f}¢  "
                     f"net-Sharpe@1¢={s1:+.3f}")
    return out


def main():
    log.info("=" * 60)
    log.info("STEP 3.4: Forecast evaluation")
    log.info("=" * 60)
    df = pd.read_parquet(FORECASTS_FILE)

    results = {}
    for factor in sorted(df["factor_removed"].unique()):
        tag = "factor_removed" if factor else "with_factor"
        results[tag] = _evaluate_factor(df[df["factor_removed"] == factor], tag)

    # Verdict
    main_un = results.get("with_factor", {}).get("unfiltered", {})
    edge = main_un.get("vs_random", {}).get("ci", (np.nan, np.nan))
    beats = np.isfinite(edge[0]) and edge[0] > 0
    verdict = ("Unfiltered BEATS random (CI excludes 0) — predictive structure present"
               if beats else
               "No variant significantly beats random — consistent with cross-domain "
               "informational efficiency (null-result protocol, roadmap 5.3)")
    log.info(f"\nVERDICT: {verdict}")
    results["verdict"] = verdict

    ensure_dirs(PHASE3_DIR)
    save_json(results, WF_METRICS_FILE)
    log.info(f"Saved metrics: {WF_METRICS_FILE}")
    return results


if __name__ == "__main__":
    main()
