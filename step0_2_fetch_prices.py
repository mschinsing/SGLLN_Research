"""
Step 0.2 — Fetch daily price histories from Polymarket CLOB API
================================================================
For every CLOB token ID found in Step 0.1, fetches the full daily price
history using the prices-history endpoint.

Input:  data/raw/markets_raw.json
Output: data/raw/prices/<token_id>.json  (one file per token)
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    CLOB_URL, PRICE_INTERVAL, PRICE_FIDELITY,
    MARKETS_RAW_FILE, PRICES_RAW_DIR,
    REQUEST_DELAY, MAX_RETRIES, RETRY_BACKOFF, REQUEST_TIMEOUT,
)
from utils import get_logger, fetch_json, save_json, load_json, ensure_dirs

log = get_logger("step0.2_fetch_prices")


def collect_all_token_ids(markets: list) -> dict:
    """
    Build a mapping: token_id -> market metadata (question, category, etc.)
    Each market may have multiple token IDs (e.g., YES and NO outcomes).
    """
    token_map = {}
    for m in markets:
        token_ids = m.get("clob_token_ids", [])
        outcomes = m.get("outcomes", "")

        # Parse outcomes — may be a JSON string like '["Yes","No"]' or a list
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except (json.JSONDecodeError, TypeError):
                outcomes = []
        if not isinstance(outcomes, list):
            outcomes = []

        for idx, tid in enumerate(token_ids):
            outcome_label = outcomes[idx] if idx < len(outcomes) else f"outcome_{idx}"
            token_map[tid] = {
                "token_id": tid,
                "condition_id": m.get("condition_id", ""),
                "question": m.get("question", ""),
                "category": m.get("category", ""),
                "outcome": outcome_label,
                "volume": m.get("volume", 0),
                "market_slug": m.get("market_slug", ""),
            }
    return token_map


def fetch_price_history(token_id: str) -> list:
    """
    Fetch daily price history for a single CLOB token ID.

    GOTCHA from API docs: the `market` param takes a CLOB token ID,
    NOT the condition ID or market slug. `fidelity` is a string token.

    Returns list of {t: unix_timestamp, p: price} dicts.
    """
    params = {
        "market": token_id,
        "interval": PRICE_INTERVAL,   # "max" = full history
        "fidelity": PRICE_FIDELITY,   # "1d" = daily
    }
    data = fetch_json(
        CLOB_URL, params=params,
        max_retries=MAX_RETRIES, backoff=RETRY_BACKOFF,
        delay=REQUEST_DELAY, timeout=REQUEST_TIMEOUT, logger=log,
    )
    # API returns {"history": [{t, p}, ...]} or just a list
    if isinstance(data, dict):
        return data.get("history", [])
    if isinstance(data, list):
        return data
    return []


def main():
    log.info("=" * 60)
    log.info("STEP 0.2: Fetch price histories from CLOB API")
    log.info("=" * 60)

    # Load markets from Step 0.1
    markets = load_json(MARKETS_RAW_FILE)
    log.info(f"Loaded {len(markets)} markets from {MARKETS_RAW_FILE}")

    # Collect all token IDs
    token_map = collect_all_token_ids(markets)
    log.info(f"Total CLOB token IDs to fetch: {len(token_map)}")

    ensure_dirs(PRICES_RAW_DIR)

    # Track progress
    success = 0
    skipped = 0
    failed = 0
    empty = 0

    for i, (tid, meta) in enumerate(token_map.items(), 1):
        out_path = os.path.join(PRICES_RAW_DIR, f"{tid}.json")

        # Skip if already fetched (resume support)
        if os.path.exists(out_path):
            skipped += 1
            if skipped % 100 == 0:
                log.info(f"  Skipped {skipped} already-fetched tokens...")
            continue

        try:
            history = fetch_price_history(tid)

            if not history:
                empty += 1
                # Save empty result to avoid re-fetching
                save_json({"token_id": tid, "meta": meta, "history": []}, out_path)
                continue

            save_json({
                "token_id": tid,
                "meta": meta,
                "history": history,
                "n_points": len(history),
            }, out_path)
            success += 1

            if success % 50 == 0:
                log.info(
                    f"  Progress: {i}/{len(token_map)} | "
                    f"success={success} empty={empty} skipped={skipped} failed={failed}"
                )

        except Exception as e:
            failed += 1
            log.warning(f"  Failed token {tid}: {e}")

    log.info(f"DONE. success={success} empty={empty} skipped={skipped} failed={failed}")
    log.info(f"Price files saved to {PRICES_RAW_DIR}/")

    return token_map


if __name__ == "__main__":
    main()
