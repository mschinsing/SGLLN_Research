"""
Step 3.5 — Calibration analysis
===============================
Tests whether contracts in LEADING clusters are better-calibrated than those in
LAGGING clusters — i.e. whether "leaders know more". Independent of the
forecasting machinery and nearly leakage-free.

  p^i_t = sigmoid(logit price);  o_i ∈ {0,1} = realized outcome
  Brier score  BS_i = (1/T_i) Σ_t (p^i_t − o_i)^2     (lower = better calibrated)

Outcome o_i is inferred from each contract's final observed probability
(> 0.5 → YES resolved). Leading vs lagging membership comes from node leadingness
L(i) = Σ_j (A_ij − A_ji) on the full-sample adjacency.

Input:  data/processed/logit_prices.parquet, phase1/adjacency_matrix.parquet,
        phase1/clusters.json
Output: data/processed/phase3/calibration.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from config import (
    LOGIT_PRICES_FILE, ADJACENCY_FILE, CLUSTERS_FILE,
    CALIBRATION_FILE, PHASE3_DIR, N_BOOTSTRAP_CI, RANDOM_SEED,
)
from utils import get_logger, ensure_dirs, save_json, load_json

log = get_logger("step3.3_calibration")
_RNG = np.random.default_rng(RANDOM_SEED)


def _brier_and_outcome(prob_col):
    """Brier score and inferred outcome for one contract's probability series."""
    p = prob_col[~np.isnan(prob_col)]
    if len(p) < 5:
        return np.nan, np.nan
    o = 1.0 if p[-1] > 0.5 else 0.0          # final price ⇒ realized outcome
    return float(np.mean((p - o) ** 2)), o


def _boot_diff(a, b):
    """Bootstrap CI on mean(a) − mean(b)."""
    draws = [a[_RNG.integers(0, len(a), len(a))].mean()
             - b[_RNG.integers(0, len(b), len(b))].mean() for _ in range(N_BOOTSTRAP_CI)]
    return float(np.mean(a) - np.mean(b)), (float(np.percentile(draws, 2.5)),
                                            float(np.percentile(draws, 97.5)))


def main():
    log.info("=" * 60)
    log.info("STEP 3.5: Calibration analysis (leaders vs laggers)")
    log.info("=" * 60)

    logit_df = pd.read_parquet(LOGIT_PRICES_FILE)
    token_ids = list(logit_df.columns)
    prob = 1.0 / (1.0 + np.exp(-logit_df.to_numpy(dtype=float)))   # sigmoid → probability

    brier = np.full(len(token_ids), np.nan)
    outcome = np.full(len(token_ids), np.nan)
    for j in range(len(token_ids)):
        brier[j], outcome[j] = _brier_and_outcome(prob[:, j])

    adj = pd.read_parquet(ADJACENCY_FILE).reindex(index=token_ids, columns=token_ids).to_numpy()
    leadingness = (adj - adj.T).sum(axis=1)               # node-level L(i)

    valid = ~np.isnan(brier)
    lead_mask = valid & (leadingness > 0)
    lag_mask = valid & (leadingness < 0)
    bs_lead = brier[lead_mask]
    bs_lag = brier[lag_mask]

    diff, ci = _boot_diff(bs_lead, bs_lag)
    _, p_mw = mannwhitneyu(bs_lead, bs_lag, alternative="less")   # leaders lower Brier?
    corr = float(np.corrcoef(leadingness[valid], brier[valid])[0, 1])

    log.info(f"  Contracts with Brier: {int(valid.sum())} "
             f"(leaders {len(bs_lead)}, laggers {len(bs_lag)})")
    log.info(f"  Mean Brier — leaders: {bs_lead.mean():.4f}  laggers: {bs_lag.mean():.4f}")
    log.info(f"  Difference (leader−lagger): {diff:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    log.info(f"  Mann–Whitney (leaders < laggers) p = {p_mw:.3f}")
    log.info(f"  corr(leadingness, Brier) = {corr:+.3f}")

    better = ci[1] < 0 and p_mw < 0.05
    verdict = ("Leaders are significantly better-calibrated (lower Brier) — supports "
               "'leaders know more'" if better else
               "No significant calibration advantage for leaders — leadership is not "
               "a calibration/accuracy edge")
    log.info(f"\nVERDICT: {verdict}")

    res = {
        "n_contracts": int(valid.sum()),
        "n_leaders": int(lead_mask.sum()), "n_laggers": int(lag_mask.sum()),
        "mean_brier_leaders": float(bs_lead.mean()), "mean_brier_laggers": float(bs_lag.mean()),
        "brier_diff_leader_minus_lagger": diff, "diff_ci": ci,
        "mannwhitney_p_leaders_lower": float(p_mw),
        "corr_leadingness_brier": corr,
        "verdict": verdict,
    }
    ensure_dirs(PHASE3_DIR)
    save_json(res, CALIBRATION_FILE)
    log.info(f"Saved calibration: {CALIBRATION_FILE}")
    return res


if __name__ == "__main__":
    main()
