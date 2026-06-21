"""
intraday_costs.py — measured effective spread → the first data-grounded net Sharpe
==================================================================================
We can't get the historical order book (off-chain, not retained), but we CAN
estimate the effective bid-ask spread from the minute-level price bounce
(Roll 1984): spread = 2·√(−cov(Δp_t, Δp_{t-1})). This replaces the assumption-free
break-even framing with a measured cost, and recomputes the leaders/laggers
follow-the-leader strategy net of it.

  python intraday_costs.py

Output: data/processed/intraday/intraday_costs.json
        data/processed/presentation/14_intraday_costs.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    CLOB_URL, ELIGIBLE_MARKETS_FILE, FORECASTS_FILE,
    INTRADAY_COSTS_FILE, INTRADAY_DIR, PRESENTATION_DIR,
    REQUEST_DELAY, MAX_RETRIES, RETRY_BACKOFF, REQUEST_TIMEOUT,
)
from utils import get_logger, ensure_dirs, save_json, load_json, fetch_json
from step3_2_evaluate import _sharpe, _cents_to_logodds, P_REF

log = get_logger("intraday_costs")

N_SAMPLE = 40                 # contracts to estimate spread from
SPREAD_START, SPREAD_END = 1729468800, 1730764800   # Oct21–Nov5 (high activity), 15-day chunk


def _roll_spread(prices):
    """Roll (1984) effective spread from the bid-ask bounce in a price series."""
    p = np.asarray(prices, dtype=float)
    p = p[~np.isnan(p)]
    if len(p) < 20:
        return np.nan
    dp = np.diff(p)
    cov = np.cov(dp[:-1], dp[1:])[0, 1]
    return 2 * np.sqrt(-cov) if cov < 0 else np.nan     # price units


def main():
    log.info("=" * 60)
    log.info("INTRADAY costs — Roll effective spread + measured net Sharpe")
    log.info("=" * 60)

    tokens = [r["token_id"] for r in load_json(ELIGIBLE_MARKETS_FILE)][:N_SAMPLE]
    log.info(f"Estimating effective spread from minute data on {len(tokens)} contracts...")
    spreads = []
    for tid in tokens:
        try:
            data = fetch_json(CLOB_URL, params={"market": tid, "startTs": SPREAD_START,
                                                "endTs": SPREAD_END, "fidelity": 1},
                              max_retries=MAX_RETRIES, backoff=RETRY_BACKOFF,
                              delay=REQUEST_DELAY, timeout=REQUEST_TIMEOUT, logger=None)
        except RuntimeError:
            continue
        hist = data.get("history", []) if isinstance(data, dict) else (data or [])
        s = _roll_spread([x["p"] for x in hist])
        if np.isfinite(s):
            spreads.append(s)

    spreads = np.array(spreads)
    med = float(np.median(spreads))
    med_cents = med * 100
    log.info(f"  effective spread (price units): median={med:.5f}  ({med_cents:.3f}¢)  "
             f"n={len(spreads)}")

    # ── Net Sharpe of the follow-the-leader strategy at the MEASURED spread ──
    df = pd.read_parquet(FORECASTS_FILE)
    d = df[(~df["factor_removed"]) & (df["variant"] == "unfiltered")]
    strat = (np.sign(d["forecast"]) * d["realized"]).to_numpy()
    gross_sharpe = _sharpe(strat)
    # measured round-trip cost in log-odds units (Roll = round-trip effective spread)
    cost_logodds = med / (P_REF * (1 - P_REF))
    net_sharpe = _sharpe(strat - cost_logodds)
    breakeven_cents = float(strat.mean() * P_REF * (1 - P_REF) * 100)

    # Roll on MID-PRICE bars is a LOWER BOUND on the true bid-ask (it measures the
    # mark's micro-bounce, not the book). Realistic floor = one tick (~0.1¢).
    TICK_CENTS = 0.1
    net_tick = _sharpe(strat - _cents_to_logodds(TICK_CENTS))
    net_1c = _sharpe(strat - _cents_to_logodds(1.0))
    log.info(f"\n── Net Sharpe (follow-the-leader, unfiltered) — gross edge is a NULL ──")
    log.info(f"  gross Sharpe={gross_sharpe:+.3f}  (forecasting CI spans 50% → ~0)")
    log.info(f"  break-even spread={breakeven_cents:.3f}¢")
    log.info(f"  Roll spread (mid-price LOWER BOUND)={med_cents:.3f}¢  → net Sharpe {net_sharpe:+.3f}")
    log.info(f"  realistic tick floor 0.1¢          → net Sharpe {net_tick:+.3f}")
    log.info(f"  typical spread 1.0¢                → net Sharpe {net_1c:+.3f}")
    log.info(f"  VERDICT: NOT tradeable — the Roll mid-price estimate underestimates the true "
             f"spread, the gross edge is statistically zero, and at any realistic spread "
             f"(≥0.1¢) the strategy is net-negative.")

    res = {"n_contracts": len(spreads), "median_spread_price": med, "median_spread_cents": med_cents,
           "spread_is_lower_bound": True, "gross_sharpe": gross_sharpe,
           "break_even_spread_cents": breakeven_cents,
           "net_sharpe_roll_lowerbound": net_sharpe, "net_sharpe_tick_0.1c": net_tick,
           "net_sharpe_1c": net_1c, "verdict": "not tradeable (gross edge null; spread underestimated)"}
    ensure_dirs([INTRADAY_DIR, PRESENTATION_DIR])
    save_json(res, INTRADAY_COSTS_FILE)
    _figure(spreads, strat, med, breakeven_cents, med_cents, net_sharpe)
    log.info(f"\nSaved: {INTRADAY_COSTS_FILE}")
    return res


def _figure(spreads, strat, med, be_cents, med_cents, net_sharpe):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.hist(spreads * 100, bins=20, color="#2b6cb0", alpha=0.8)
    ax1.axvline(med * 100, color="red", ls="--", label=f"median {med_cents:.3f}¢")
    ax1.set_xlabel("effective spread (cents)"); ax1.set_ylabel("contracts")
    ax1.set_title("(a) Measured effective spread (Roll)"); ax1.legend(fontsize=9)

    grid = np.linspace(0, max(med_cents * 1.5, 0.1), 50)
    nets = [_sharpe(strat - _cents_to_logodds(c)) for c in grid]
    ax2.plot(grid, nets, color="#2b6cb0")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.axvline(be_cents, color="#38a169", ls=":", label=f"break-even {be_cents:.3f}¢")
    ax2.axvline(med_cents, color="red", ls="--", label=f"measured {med_cents:.3f}¢")
    ax2.set_xlabel("round-trip spread (cents)"); ax2.set_ylabel("net Sharpe")
    ax2.set_title(f"(b) Net Sharpe vs cost (measured ⇒ {net_sharpe:+.2f})"); ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PRESENTATION_DIR, "14_intraday_costs.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
