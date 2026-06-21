"""
Step 0.3 — Fetch trade data for liquidity metadata
====================================================
For each eligible market (by condition_id), fetches trade records from the
Polymarket data API. Aggregates into daily liquidity metrics:
  - daily volume (USD)
  - daily trade count
  - daily unique-maker count (proxy for bid-ask activity)

Input:  data/raw/markets_raw.json
Output: data/raw/trades/<condition_id>.json  (one file per market)
        data/raw/daily_volumes.parquet       (aggregated)
"""
import sys, os, json
from datetime import datetime, timezone
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DATA_URL, MARKETS_RAW_FILE, TRADES_RAW_DIR, VOLUME_FILE,
    START_DATE, END_DATE,
    REQUEST_DELAY, MAX_RETRIES, RETRY_BACKOFF, REQUEST_TIMEOUT,
)
from utils import get_logger, fetch_json, save_json, load_json, ensure_dirs

log = get_logger("step0.3_fetch_trades")

TRADES_PAGE_LIMIT = 1000  # API max per request
MAX_TRADE_PAGES = 200     # Safety cap per market (200k trades)


def fetch_trades_for_market(condition_id: str) -> list:
    """
    Paginate through the trades endpoint for a given conditionId.
    Returns list of trade records.
    """
    all_trades = []
    offset = 0

    for page in range(1, MAX_TRADE_PAGES + 1):
        params = {
            "market": condition_id,
            "limit": TRADES_PAGE_LIMIT,
            "offset": offset,
        }

        try:
            data = fetch_json(
                DATA_URL, params=params,
                max_retries=MAX_RETRIES, backoff=RETRY_BACKOFF,
                delay=REQUEST_DELAY, timeout=REQUEST_TIMEOUT, logger=log,
            )
        except RuntimeError:
            log.warning(f"  Gave up fetching trades for {condition_id} at offset {offset}")
            break

        if not data:
            break

        all_trades.extend(data)
        offset += TRADES_PAGE_LIMIT

        if len(data) < TRADES_PAGE_LIMIT:
            break

    return all_trades


def aggregate_daily_liquidity(trades: list, condition_id: str) -> list:
    """
    Aggregate trades into daily liquidity metrics.
    Returns list of dicts with: date, volume_usd, trade_count, unique_makers.
    """
    daily = defaultdict(lambda: {"volume_usd": 0.0, "trade_count": 0, "makers": set()})

    for t in trades:
        # Parse timestamp — trades may have 'timestamp' or 'createdAt'
        ts_raw = t.get("timestamp") or t.get("createdAt") or t.get("matchTime", "")
        if not ts_raw:
            continue

        try:
            if isinstance(ts_raw, (int, float)):
                dt = datetime.fromtimestamp(ts_raw, tz=timezone.utc).date()
            else:
                # Try ISO format
                dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).date()
        except (ValueError, OSError):
            continue

        day_str = dt.isoformat()
        size = float(t.get("size", 0) or 0)
        price = float(t.get("price", 0) or 0)
        volume = size * price  # rough USD estimate

        daily[day_str]["volume_usd"] += volume
        daily[day_str]["trade_count"] += 1

        maker = t.get("maker", t.get("makerAddress", ""))
        if maker:
            daily[day_str]["makers"].add(maker)

    result = []
    for day_str, metrics in sorted(daily.items()):
        result.append({
            "condition_id": condition_id,
            "date": day_str,
            "volume_usd": round(metrics["volume_usd"], 2),
            "trade_count": metrics["trade_count"],
            "unique_makers": len(metrics["makers"]),
        })

    return result


def main():
    log.info("=" * 60)
    log.info("STEP 0.3: Fetch trade data for liquidity metadata")
    log.info("=" * 60)

    markets = load_json(MARKETS_RAW_FILE)
    log.info(f"Loaded {len(markets)} markets")

    # De-duplicate by condition_id
    cid_map = {}
    for m in markets:
        cid = m.get("condition_id", "")
        if cid and cid not in cid_map:
            cid_map[cid] = m

    log.info(f"Unique condition IDs: {len(cid_map)}")
    ensure_dirs(TRADES_RAW_DIR)

    all_daily_rows = []
    success = 0
    skipped = 0
    failed = 0

    for i, (cid, meta) in enumerate(cid_map.items(), 1):
        out_path = os.path.join(TRADES_RAW_DIR, f"{cid}.json")

        # Resume support
        if os.path.exists(out_path):
            skipped += 1
            # Still load for aggregation
            try:
                cached = load_json(out_path)
                all_daily_rows.extend(cached.get("daily_liquidity", []))
            except Exception:
                pass
            continue

        try:
            trades = fetch_trades_for_market(cid)
            daily = aggregate_daily_liquidity(trades, cid)

            save_json({
                "condition_id": cid,
                "question": meta.get("question", ""),
                "total_trades": len(trades),
                "daily_liquidity": daily,
            }, out_path)

            all_daily_rows.extend(daily)
            success += 1

            if success % 25 == 0:
                log.info(
                    f"  Progress: {i}/{len(cid_map)} | "
                    f"success={success} skipped={skipped} failed={failed} | "
                    f"trades for last market: {len(trades)}"
                )

        except Exception as e:
            failed += 1
            log.warning(f"  Failed {cid}: {e}")

    # Save aggregated daily volumes as parquet
    if all_daily_rows:
        try:
            import pandas as pd
            df = pd.DataFrame(all_daily_rows)
            df["date"] = pd.to_datetime(df["date"])
            ensure_dirs(os.path.dirname(VOLUME_FILE))
            df.to_parquet(VOLUME_FILE, index=False)
            log.info(f"Saved aggregated daily volumes: {VOLUME_FILE} ({len(df)} rows)")
        except ImportError:
            # Fallback: save as JSON
            fallback = VOLUME_FILE.replace(".parquet", ".json")
            save_json(all_daily_rows, fallback)
            log.info(f"Saved aggregated daily volumes (JSON fallback): {fallback}")

    log.info(f"DONE. success={success} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
