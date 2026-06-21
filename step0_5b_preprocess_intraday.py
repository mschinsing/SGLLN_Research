"""
Step 0.5b — Preprocess INTRADAY (hourly) prices → returns matrix
================================================================
Mirror of step0_5_preprocess.py on hourly bars. Aligns each eligible token's
hourly price history to a common hourly index, clips→logit→returns, winsorizes
in-window, forward-fills within active spans. Same 354-contract universe and
token order as the daily returns (for direct comparison).

Input:  data/raw/prices_hourly/<token>.json, data/processed/eligible_markets.json
Output: data/processed/intraday/returns_hourly.parquet   (n_hours × 354)
"""
import sys, os
from datetime import datetime, timedelta, timezone
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from config import (
    START_DATE, END_DATE, CLIP_LO, CLIP_HI, WINSOR_LO, WINSOR_HI, RESOLUTION_EXCLUDE_H,
    PRICES_HOURLY_DIR, ELIGIBLE_MARKETS_FILE, RETURNS_HOURLY_FILE, INTRADAY_DIR,
)
from utils import get_logger, load_json, ensure_dirs
from step0_5_preprocess import clip_and_logit, compute_returns, winsorize

log = get_logger("step0.5b_preprocess_intraday")


def _hour_index():
    start = datetime(START_DATE.year, START_DATE.month, START_DATE.day, tzinfo=timezone.utc)
    end = datetime(END_DATE.year, END_DATE.month, END_DATE.day, tzinfo=timezone.utc) + timedelta(days=1)
    return pd.date_range(start, end, freq="h", inclusive="left", tz="UTC")


def main():
    log.info("=" * 60)
    log.info("STEP 0.5b: Preprocess intraday (hourly) → returns matrix")
    log.info("=" * 60)

    eligible = load_json(ELIGIBLE_MARKETS_FILE)
    token_ids = [r["token_id"] for r in eligible]
    end_dates = {r["token_id"]: r.get("end_date", "") for r in eligible}
    hours = _hour_index()
    ts_to_idx = {int(t.timestamp()): i for i, t in enumerate(hours)}
    n_h, n_t = len(hours), len(token_ids)
    log.info(f"Hourly index: {hours[0]} → {hours[-1]} ({n_h} hours) × {n_t} contracts")

    prices = np.full((n_h, n_t), np.nan)
    fetched = 0
    for col, tid in enumerate(token_ids):
        path = os.path.join(PRICES_HOURLY_DIR, f"{tid}.json")
        if not os.path.exists(path):
            continue
        hist = load_json(path).get("history", [])
        if hist:
            fetched += 1
        for pt in hist:
            idx = ts_to_idx.get((int(pt["t"]) // 3600) * 3600)   # floor to hour bucket
            if idx is not None and 0 < float(pt["p"]) < 1:
                prices[idx, col] = float(pt["p"])

        # resolution exclusion: drop final RESOLUTION_EXCLUDE_H hours before end
        ed = end_dates.get(tid, "")
        if ed:
            try:
                edt = datetime.fromisoformat(ed.replace("Z", "+00:00"))
                if edt.tzinfo is None:
                    edt = edt.replace(tzinfo=timezone.utc)
                cutoff = edt - timedelta(hours=RESOLUTION_EXCLUDE_H)
                prices[hours > cutoff, col] = np.nan
            except (ValueError, TypeError):
                pass

        # forward-fill within active span (causal)
        last = np.nan
        for i in range(n_h):
            if not np.isnan(prices[i, col]):
                last = prices[i, col]
            elif not np.isnan(last):
                prices[i, col] = last

    log.info(f"Contracts with hourly data: {fetched}/{n_t}")

    logits = clip_and_logit(prices)
    returns = winsorize(compute_returns(logits), WINSOR_LO, WINSOR_HI)

    ensure_dirs(INTRADAY_DIR)
    idx = [t.isoformat() for t in hours]
    df = pd.DataFrame(returns, index=idx, columns=token_ids)
    df.index.name = "hour"
    df.to_parquet(RETURNS_HOURLY_FILE)
    log.info(f"Saved hourly returns: {RETURNS_HOURLY_FILE}  shape={returns.shape}")
    log.info(f"  non-NaN returns: {np.sum(~np.isnan(returns)):,} "
             f"({100*np.mean(~np.isnan(returns)):.1f}% filled)")
    return df


if __name__ == "__main__":
    main()
