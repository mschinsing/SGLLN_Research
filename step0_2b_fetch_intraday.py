"""
Step 0.2b — Fetch INTRADAY price histories (parallel track)
===========================================================
The CLOB /prices-history endpoint returns empty for `interval=max&fidelity=60`,
but returns full intraday data when queried with explicit `startTs`/`endTs` in
<=15-day chunks. This fetches hourly (or minute) bars for the daily-eligible
tokens over the study window, chunk-by-chunk, and concatenates.

Does NOT touch the daily pipeline — writes to data/raw/prices_hourly/.

Usage:
    python step0_2b_fetch_intraday.py --limit 5      # validate on 5 tokens first
    python step0_2b_fetch_intraday.py                # full fetch (resumable)
    python step0_2b_fetch_intraday.py --fidelity 1   # minute bars (for spreads)

Output: data/raw/prices_hourly/<token>.json  (one file per token)
"""
import sys, os, argparse
from datetime import datetime, timezone, timedelta
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    CLOB_URL, PRICES_HOURLY_DIR, ELIGIBLE_MARKETS_FILE,
    START_DATE, END_DATE, INTRADAY_FIDELITY, INTRADAY_CHUNK_DAYS,
    REQUEST_DELAY, MAX_RETRIES, RETRY_BACKOFF, REQUEST_TIMEOUT,
)
from utils import get_logger, fetch_json, save_json, load_json, ensure_dirs

log = get_logger("step0.2b_fetch_intraday")


def _chunks(start_date, end_date, chunk_days):
    """Yield (startTs, endTs) unix-second pairs covering [start, end] in chunks."""
    s = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(days=1)
    step = timedelta(days=chunk_days)
    while s < end:
        e = min(s + step, end)
        yield int(s.timestamp()), int(e.timestamp())
        s = e


def fetch_token_intraday(token_id, fidelity):
    """Fetch all intraday bars for one token over the window via chunked requests."""
    by_ts = {}
    for cs, ce in _chunks(START_DATE, END_DATE, INTRADAY_CHUNK_DAYS):
        try:
            data = fetch_json(
                CLOB_URL, params={"market": token_id, "startTs": cs, "endTs": ce, "fidelity": fidelity},
                max_retries=MAX_RETRIES, backoff=RETRY_BACKOFF,
                delay=REQUEST_DELAY, timeout=REQUEST_TIMEOUT, logger=None,
            )
        except RuntimeError:
            continue                                    # out-of-range chunk → skip
        hist = data.get("history", []) if isinstance(data, dict) else (data or [])
        for pt in hist:
            if pt.get("t") is not None and pt.get("p") is not None:
                by_ts[pt["t"]] = pt                     # dedupe overlapping chunk edges
    return [by_ts[t] for t in sorted(by_ts)]


def main():
    ap = argparse.ArgumentParser(description="Fetch intraday prices for eligible tokens")
    ap.add_argument("--limit", type=int, default=None, help="only first N tokens (validation)")
    ap.add_argument("--fidelity", type=int, default=INTRADAY_FIDELITY, help="minutes/bar (60=hourly,1=minute)")
    args = ap.parse_args()

    log.info("=" * 60)
    log.info(f"STEP 0.2b: Fetch intraday prices (fidelity={args.fidelity} min/bar)")
    log.info(f"Window: {START_DATE} -> {END_DATE}  | {INTRADAY_CHUNK_DAYS}-day chunks")
    log.info("=" * 60)

    eligible = load_json(ELIGIBLE_MARKETS_FILE)
    tokens = [r["token_id"] for r in eligible]
    if args.limit:
        tokens = tokens[:args.limit]
    ensure_dirs(PRICES_HOURLY_DIR)
    log.info(f"Tokens to fetch: {len(tokens)}")

    success = skipped = empty = 0
    for i, tid in enumerate(tokens, 1):
        out = os.path.join(PRICES_HOURLY_DIR, f"{tid}.json")
        if os.path.exists(out):
            skipped += 1
            continue
        hist = fetch_token_intraday(tid, args.fidelity)
        save_json({"token_id": tid, "fidelity": args.fidelity, "n_points": len(hist),
                   "history": hist}, out)
        if hist:
            success += 1
        else:
            empty += 1
        if i % 25 == 0 or args.limit:
            log.info(f"  {i}/{len(tokens)}: {tid[:14]}… -> {len(hist)} pts "
                     f"(success={success} empty={empty} skipped={skipped})")

    log.info(f"DONE. success={success} empty={empty} skipped={skipped} -> {PRICES_HOURLY_DIR}/")


if __name__ == "__main__":
    main()
