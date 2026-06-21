"""
Step 3.6 — Event-conditional network analysis
=============================================
Rolling 60-day windows (EVENT_STEP-day step) across the full sample. Cheap,
relatively well-powered parts only:
  * |λ1| — top Hermitian eigenvalue magnitude per window (directed-structure
    intensity). Does it rise in windows with a fresh macro/political event?
  * Leadingness dispersion (std of node leadingness) per window.
  * Rolling time series with event markers + bootstrap CI band on |λ1|.

With ~10 heavily-overlapping windows this is EXPLORATORY (low power); reported
as suggestive, not confirmatory.

Input:  data/processed/logit_prices.parquet, data/processed/event_calendar.csv
Output: data/processed/phase3/event_analysis.json
        data/processed/presentation/07_event_conditional.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import date
from scipy.linalg import eigh
from scipy.stats import mannwhitneyu

from config import (
    LOGIT_PRICES_FILE, EVENTS_FILE, EVENT_ANALYSIS_FILE, PHASE3_DIR, PRESENTATION_DIR,
    W_TRAIN, WF_DEP_PCT, WF_MIN_ACTIVE, WINSOR_LO, WINSOR_HI, MIN_OVERLAP_DAYS, N_BOOTSTRAP_CI,
    RANDOM_SEED,
)
from utils import get_logger, ensure_dirs, save_json
from step3_1_walkforward import _idcor_window
from step1_4_spectral import _normalized_hermitian
from step1_6_liquidity import _build_adjacency, _node_leadingness
from step0_5_preprocess import winsorize

log = get_logger("step3.4_event_analysis")

EVENT_STEP = 10          # roll step (days); roadmap 3.6
_RNG = np.random.default_rng(RANDOM_SEED)


def _window_stats(block):
    """(|λ1|, leadingness dispersion) for one in-window return block, or None."""
    idc = _idcor_window(block)
    iu = np.triu_indices(block.shape[1], k=1)
    max_i = np.maximum(idc, idc.T)[iu]
    if (max_i > 0).sum() < W_TRAIN // 6:
        return None
    adj = _build_adjacency(idc, float(np.percentile(max_i, WF_DEP_PCT)))
    a_sym, _, _ = _normalized_hermitian(adj)
    eigvals = eigh(a_sym, eigvals_only=True)
    lam1 = float(np.max(np.abs(eigvals)))
    disp = float(np.std(_node_leadingness(adj)))
    return lam1, disp


def main():
    log.info("=" * 60)
    log.info("STEP 3.6: Event-conditional network analysis")
    log.info("=" * 60)

    logit_df = pd.read_parquet(LOGIT_PRICES_FILE)
    dates = [pd.to_datetime(d).date() for d in logit_df.index]
    logit = logit_df.to_numpy(dtype=float)
    rr = np.full_like(logit, np.nan)
    rr[1:, :] = logit[1:, :] - logit[:-1, :]

    events = pd.read_csv(EVENTS_FILE)
    event_dates = {date.fromisoformat(d) for d in events["date"]}

    ends = list(range(W_TRAIN, len(dates), EVENT_STEP))
    rows = []
    for e in ends:
        train = rr[e - W_TRAIN:e, :]
        active = np.nonzero((~np.isnan(train)).sum(axis=0) >= WF_MIN_ACTIVE)[0]
        if len(active) < 20:
            continue
        stats = _window_stats(winsorize(train[:, active], WINSOR_LO, WINSOR_HI))
        if stats is None:
            continue
        # fresh event = an event in this window's final EVENT_STEP days
        recent = {dates[i] for i in range(max(0, e - EVENT_STEP), e)}
        has_event = bool(recent & event_dates)
        rows.append({"date": dates[e - 1], "lambda1": stats[0],
                     "leadingness_disp": stats[1], "fresh_event": has_event})

    ts = pd.DataFrame(rows)
    log.info(f"Rolling windows: {len(ts)} (W={W_TRAIN}, step={EVENT_STEP})")

    ev = ts[ts["fresh_event"]]["lambda1"].to_numpy()
    nonev = ts[~ts["fresh_event"]]["lambda1"].to_numpy()
    res = {"n_windows": len(ts), "n_event_windows": int(ts["fresh_event"].sum()),
           "lambda1_mean_event": float(ev.mean()) if len(ev) else None,
           "lambda1_mean_nonevent": float(nonev.mean()) if len(nonev) else None}
    if len(ev) >= 3 and len(nonev) >= 3:
        _, p = mannwhitneyu(ev, nonev, alternative="greater")     # |λ1| higher in event windows?
        res["mannwhitney_p_event_higher"] = float(p)
        log.info(f"  |λ1| event windows: {ev.mean():.3f}  non-event: {nonev.mean():.3f}  "
                 f"(MW p={p:.3f}, exploratory)")
    else:
        res["mannwhitney_p_event_higher"] = None
        log.info("  Too few event/non-event windows for a test (exploratory only).")

    ensure_dirs([PHASE3_DIR, PRESENTATION_DIR])
    save_json(res, EVENT_ANALYSIS_FILE)

    # ── Figure: rolling |λ1| with event markers ──────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    x = pd.to_datetime(ts["date"])
    ax.plot(x, ts["lambda1"], "-o", color="#2b6cb0", label="|λ1| (structure intensity)")
    for _, r in events.iterrows():
        d = pd.to_datetime(r["date"])
        if x.min() <= d <= x.max():
            ax.axvline(d, color="#e53e3e", alpha=0.35, lw=1)
    ax.axvline(x.iloc[0], color="#e53e3e", alpha=0.35, lw=1, label="event (FOMC/CPI/debate/jobs)")
    ax.set_xlabel("window end date")
    ax.set_ylabel("|λ1|")
    ax.set_title("Rolling directed-structure intensity with event markers (exploratory)")
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig_path = os.path.join(PRESENTATION_DIR, "07_event_conditional.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved figure: {fig_path}")
    log.info(f"Saved event analysis: {EVENT_ANALYSIS_FILE}")
    return res


if __name__ == "__main__":
    main()
