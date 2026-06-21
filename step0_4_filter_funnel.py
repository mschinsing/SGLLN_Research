"""
Step 0.4 — Filtering funnel
=============================
Applies the four eligibility criteria from the proposal (§2) and produces:
  1. A filtering-funnel table (count dropped at each step)
  2. The list of eligible markets/token IDs for downstream analysis

Input:  data/raw/markets_raw.json, data/raw/prices/*.json
Output: data/processed/filter_funnel.csv
        data/processed/eligible_markets.json
        data/processed/eligible_token_ids.json
"""
import sys, os, json, glob
from datetime import datetime, date, timedelta, timezone
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from config import (
    START_DATE, END_DATE, TOTAL_DAYS,
    MIN_ACTIVE_DAYS, MIN_VOLUME_USD, MIN_RETURN_STD, MAX_GAP_DAYS,
    RESOLUTION_EXCLUDE_H, CLIP_LO, CLIP_HI,
    CATEGORY_ALIASES, MACRO_CATEGORY, MACRO_KEYWORDS,
    MARKETS_RAW_FILE, PRICES_RAW_DIR,
    FILTER_FUNNEL_FILE, ELIGIBLE_MARKETS_FILE, ELIGIBLE_IDS_FILE,
    PROCESSED_DIR,
)
from utils import get_logger, load_json, save_json, ensure_dirs

log = get_logger("step0.4_filter_funnel")


def normalize_category(category: str, question: str) -> str:
    """Fold straggler tags into canonical categories (CATEGORY_ALIASES) and
    reclassify macro/monetary-policy markets (Fed, rates, shutdown) out of
    Politics into MACRO_CATEGORY based on the question text."""
    cat = CATEGORY_ALIASES.get(category, category)
    ql = (question or "").lower()
    if any(kw in ql for kw in MACRO_KEYWORDS):
        return MACRO_CATEGORY
    return cat


def load_price_history(token_id: str) -> list:
    """Load price history for a single token ID."""
    path = os.path.join(PRICES_RAW_DIR, f"{token_id}.json")
    if not os.path.exists(path):
        return []
    data = load_json(path)
    return data.get("history", [])


def history_to_daily_series(history: list) -> dict:
    """
    Convert raw price history to a date -> price mapping.
    Filters to our analysis window [START_DATE, END_DATE].
    Returns dict: {date_str: price}
    """
    daily = {}
    for point in history:
        t = point.get("t")
        p = point.get("p")
        if t is None or p is None:
            continue

        # Parse timestamp
        try:
            if isinstance(t, (int, float)):
                dt = datetime.fromtimestamp(t, tz=timezone.utc).date()
            else:
                dt = datetime.fromisoformat(str(t).replace("Z", "+00:00")).date()
        except (ValueError, OSError):
            continue

        # Filter to window
        if dt < START_DATE or dt > END_DATE:
            continue

        p = float(p)
        if 0 < p < 1:
            daily[dt.isoformat()] = p

    return daily


def compute_active_days(daily: dict) -> int:
    """Count distinct dates with a price observation."""
    return len(daily)


def compute_max_gap(daily: dict) -> int:
    """
    Compute the maximum gap (in calendar days) between consecutive
    price observations within the analysis window.
    """
    if len(daily) < 2:
        return TOTAL_DAYS

    dates = sorted(datetime.fromisoformat(d).date() for d in daily.keys())
    max_gap = 0
    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i - 1]).days
        if gap > max_gap:
            max_gap = gap
    return max_gap


def compute_logit_return_std(daily: dict) -> float:
    """
    Compute std dev of daily log-odds returns.
    Clips prices to [CLIP_LO, CLIP_HI] before logit transform.
    """
    if len(daily) < 3:
        return 0.0

    dates = sorted(daily.keys())
    prices = [daily[d] for d in dates]

    # Clip
    prices = [max(CLIP_LO, min(CLIP_HI, p)) for p in prices]

    # Logit transform
    logits = [np.log(p / (1 - p)) for p in prices]

    # Daily returns
    returns = [logits[i] - logits[i - 1] for i in range(1, len(logits))]

    if not returns:
        return 0.0

    return float(np.std(returns))


def exclude_resolution_window(daily: dict, end_date_str: str) -> dict:
    """
    Remove observations in the final RESOLUTION_EXCLUDE_H hours
    before the market's resolution/end date.
    """
    if not end_date_str:
        return daily

    try:
        end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        cutoff = (end_dt - timedelta(hours=RESOLUTION_EXCLUDE_H)).date()
    except (ValueError, TypeError):
        return daily

    return {d: p for d, p in daily.items()
            if datetime.fromisoformat(d).date() <= cutoff}


def main():
    log.info("=" * 60)
    log.info("STEP 0.4: Filtering funnel")
    log.info("=" * 60)

    markets = load_json(MARKETS_RAW_FILE)
    log.info(f"Loaded {len(markets)} markets")

    # ── Build per-token records ──────────────────────────────────────
    # Every Polymarket market here is binary (exactly 2 outcomes), and the two
    # outcome tokens are perfectly anti-correlated (price_1 = 1 - price_0, so
    # their log-odds returns are exact negatives). Keeping both would inject
    # -1 correlations into the lead-lag network and near-duplicate embeddings,
    # so we keep only the first outcome (index 0) as the market's representative.
    records = []
    for m in markets:
        token_ids = m.get("clob_token_ids", [])
        for idx, tid in enumerate(token_ids):
            if idx != 0:
                continue
            records.append({
                "token_id": tid,
                "condition_id": m.get("condition_id", ""),
                "question": m.get("question", ""),
                "category": normalize_category(m.get("category", ""), m.get("question", "")),
                "volume": m.get("volume", 0),
                "end_date": m.get("end_date", ""),
                "outcome_index": idx,
                "market_slug": m.get("market_slug", ""),
                "description": m.get("description", ""),
            })

    log.info(f"Total token-level records: {len(records)}")

    # ── Funnel tracking ──────────────────────────────────────────────
    funnel = []
    funnel.append(("0_total_tokens", len(records)))

    # Step F1: Must have price data
    for rec in records:
        history = load_price_history(rec["token_id"])
        daily_raw = history_to_daily_series(history)
        # Exclude resolution window
        daily = exclude_resolution_window(daily_raw, rec.get("end_date", ""))
        rec["daily"] = daily
        rec["active_days"] = compute_active_days(daily)
        rec["max_gap"] = compute_max_gap(daily)
        rec["return_std"] = compute_logit_return_std(daily)

    has_data = [r for r in records if r["active_days"] > 0]
    funnel.append(("1_has_price_data", len(has_data)))
    log.info(f"After price-data check: {len(has_data)}")

    # Step F2: Active >= MIN_ACTIVE_DAYS
    active_enough = [r for r in has_data if r["active_days"] >= MIN_ACTIVE_DAYS]
    funnel.append(("2_active_days_ge_60", len(active_enough)))
    log.info(f"After active-days filter (>={MIN_ACTIVE_DAYS}): {len(active_enough)}")

    # Step F3: Volume >= MIN_VOLUME_USD
    vol_enough = [r for r in active_enough if r["volume"] >= MIN_VOLUME_USD]
    funnel.append(("3_volume_ge_50k", len(vol_enough)))
    log.info(f"After volume filter (>=${MIN_VOLUME_USD:,}): {len(vol_enough)}")

    # Step F4: Return std > MIN_RETURN_STD
    var_enough = [r for r in vol_enough if r["return_std"] > MIN_RETURN_STD]
    funnel.append(("4_return_std_gt_001", len(var_enough)))
    log.info(f"After return-std filter (>{MIN_RETURN_STD}): {len(var_enough)}")

    # Step F5: Max gap <= MAX_GAP_DAYS
    gap_ok = [r for r in var_enough if r["max_gap"] <= MAX_GAP_DAYS]
    funnel.append(("5_max_gap_le_5", len(gap_ok)))
    log.info(f"After max-gap filter (<={MAX_GAP_DAYS} days): {len(gap_ok)}")

    eligible = gap_ok

    # ── Summary stats ────────────────────────────────────────────────
    categories = {r["category"] for r in eligible if r["category"]}
    log.info(f"\nELIGIBLE: {len(eligible)} tokens across {len(categories)} categories")
    for cat in sorted(categories):
        count = sum(1 for r in eligible if r["category"] == cat)
        log.info(f"  {cat}: {count}")

    # ── Save funnel table ────────────────────────────────────────────
    ensure_dirs(PROCESSED_DIR)

    import csv
    with open(FILTER_FUNNEL_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "count", "dropped"])
        prev = funnel[0][1]
        for step_name, count in funnel:
            writer.writerow([step_name, count, prev - count])
            prev = count
    log.info(f"Saved funnel table: {FILTER_FUNNEL_FILE}")

    # ── Save eligible markets ────────────────────────────────────────
    eligible_out = []
    for r in eligible:
        eligible_out.append({
            "token_id": r["token_id"],
            "condition_id": r["condition_id"],
            "question": r["question"],
            "category": r["category"],
            "volume": r["volume"],
            "active_days": r["active_days"],
            "max_gap": r["max_gap"],
            "return_std": r["return_std"],
            "outcome_index": r["outcome_index"],
            "market_slug": r["market_slug"],
            "description": r["description"],
        })
    save_json(eligible_out, ELIGIBLE_MARKETS_FILE)
    log.info(f"Saved eligible markets: {ELIGIBLE_MARKETS_FILE}")

    # Save just the token IDs for quick loading
    save_json([r["token_id"] for r in eligible], ELIGIBLE_IDS_FILE)
    log.info(f"Saved eligible token IDs: {ELIGIBLE_IDS_FILE}")

    return eligible


if __name__ == "__main__":
    main()
