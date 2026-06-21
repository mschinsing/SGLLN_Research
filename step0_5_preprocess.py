"""
Step 0.5 — Preprocessing: logit prices, returns, winsorization, overlap matrix
===============================================================================
Takes the eligible token list and raw price data, produces:
  1. Logit-transformed price matrix (date × token)
  2. Daily returns matrix (winsorized)
  3. Pairwise overlap matrix (number of co-active days)
  4. Liquidity metadata table

Input:  data/processed/eligible_markets.json, data/raw/prices/*.json
Output: data/processed/logit_prices.parquet
        data/processed/returns.parquet
        data/processed/overlap_matrix.parquet
        data/processed/liquidity_metadata.parquet
"""
import sys, os, json
from datetime import datetime, date, timedelta, timezone
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from config import (
    START_DATE, END_DATE, TOTAL_DAYS,
    CLIP_LO, CLIP_HI, WINSOR_LO, WINSOR_HI,
    MIN_OVERLAP_DAYS, RESOLUTION_EXCLUDE_H,
    PRICES_RAW_DIR, ELIGIBLE_MARKETS_FILE,
    LOGIT_PRICES_FILE, RETURNS_FILE, OVERLAP_MATRIX_FILE,
    LIQUIDITY_FILE, VOLUME_FILE, PROCESSED_DIR,
)
from utils import get_logger, load_json, ensure_dirs

log = get_logger("step0.5_preprocess")


def build_date_index() -> list:
    """Generate all calendar dates in [START_DATE, END_DATE]."""
    dates = []
    d = START_DATE
    while d <= END_DATE:
        dates.append(d)
        d += timedelta(days=1)
    return dates


def load_and_align_prices(eligible: list, date_index: list) -> tuple:
    """
    Load raw price histories for eligible tokens and align to a common
    date index. Forward-fills gaps within active periods.

    Returns:
        price_matrix: np.ndarray of shape (n_dates, n_tokens), NaN where inactive
        token_ids: list of token ID strings
    """
    n_dates = len(date_index)
    n_tokens = len(eligible)
    date_to_idx = {d: i for i, d in enumerate(date_index)}

    price_matrix = np.full((n_dates, n_tokens), np.nan)
    token_ids = []

    for col, rec in enumerate(eligible):
        tid = rec["token_id"]
        token_ids.append(tid)

        path = os.path.join(PRICES_RAW_DIR, f"{tid}.json")
        if not os.path.exists(path):
            continue

        data = load_json(path)
        history = data.get("history", [])

        for point in history:
            t = point.get("t")
            p = point.get("p")
            if t is None or p is None:
                continue

            try:
                if isinstance(t, (int, float)):
                    dt = datetime.fromtimestamp(t, tz=timezone.utc).date()
                else:
                    dt = datetime.fromisoformat(str(t).replace("Z", "+00:00")).date()
            except (ValueError, OSError):
                continue

            if dt in date_to_idx:
                price_matrix[date_to_idx[dt], col] = float(p)

        # Exclude resolution window
        end_date_str = rec.get("end_date", "")
        if end_date_str:
            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)
                cutoff = (end_dt - timedelta(hours=RESOLUTION_EXCLUDE_H)).date()
                for i, d in enumerate(date_index):
                    if d > cutoff:
                        price_matrix[i, col] = np.nan
            except (ValueError, TypeError):
                pass

        # Forward-fill within active periods
        last_valid = np.nan
        for i in range(n_dates):
            if not np.isnan(price_matrix[i, col]):
                last_valid = price_matrix[i, col]
            elif not np.isnan(last_valid):
                # Only forward-fill if we've seen at least one observation
                price_matrix[i, col] = last_valid

    return price_matrix, token_ids


def clip_and_logit(prices: np.ndarray) -> np.ndarray:
    """
    Clip prices to [CLIP_LO, CLIP_HI] and apply logit transform.
    x = log(p / (1 - p))
    """
    clipped = np.clip(prices, CLIP_LO, CLIP_HI)
    # Preserve NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        logits = np.log(clipped / (1 - clipped))
    logits[np.isnan(prices)] = np.nan
    return logits


def compute_returns(logits: np.ndarray) -> np.ndarray:
    """
    Daily log-odds returns: r_t = x_t - x_{t-1}
    First row is NaN (no prior day).
    """
    returns = np.full_like(logits, np.nan)
    returns[1:, :] = logits[1:, :] - logits[:-1, :]
    return returns


def winsorize(returns: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Winsorize each column at the lo/hi percentiles.
    Operates column-by-column to respect per-contract distributions.
    """
    result = returns.copy()
    n_cols = result.shape[1]
    for col in range(n_cols):
        col_data = result[:, col]
        valid = col_data[~np.isnan(col_data)]
        if len(valid) < 5:
            continue
        lo_val = np.percentile(valid, lo * 100)
        hi_val = np.percentile(valid, hi * 100)
        col_data = np.where(np.isnan(col_data), col_data,
                            np.clip(col_data, lo_val, hi_val))
        result[:, col] = col_data
    return result


def compute_overlap_matrix(returns: np.ndarray) -> np.ndarray:
    """
    Compute pairwise overlap: number of days where BOTH tokens have
    non-NaN returns. Shape: (n_tokens, n_tokens).
    """
    valid = ~np.isnan(returns)  # (n_dates, n_tokens)
    # Overlap = valid_i^T @ valid_j (dot product of boolean columns)
    overlap = valid.astype(np.float64).T @ valid.astype(np.float64)
    return overlap.astype(int)


def build_liquidity_metadata(eligible: list, returns: np.ndarray) -> list:
    """
    Build per-token liquidity metadata table.
    """
    records = []
    for col, rec in enumerate(eligible):
        col_returns = returns[:, col]
        valid = col_returns[~np.isnan(col_returns)]
        records.append({
            "token_id": rec["token_id"],
            "condition_id": rec.get("condition_id", ""),
            "question": rec.get("question", ""),
            "category": rec.get("category", ""),
            "volume": rec.get("volume", 0),
            "active_days": rec.get("active_days", 0),
            "return_std": float(np.std(valid)) if len(valid) > 0 else 0,
            "return_mean": float(np.mean(valid)) if len(valid) > 0 else 0,
            "n_valid_returns": int(len(valid)),
        })
    return records


def main():
    log.info("=" * 60)
    log.info("STEP 0.5: Preprocessing — logit prices, returns, overlap")
    log.info("=" * 60)

    # Load eligible markets
    eligible = load_json(ELIGIBLE_MARKETS_FILE)
    log.info(f"Eligible tokens: {len(eligible)}")

    # Build date index
    date_index = build_date_index()
    log.info(f"Date index: {date_index[0]} to {date_index[-1]} ({len(date_index)} days)")

    # Load and align prices
    log.info("Loading and aligning price histories...")
    prices, token_ids = load_and_align_prices(eligible, date_index)
    log.info(f"Price matrix shape: {prices.shape}")

    # Clip and logit transform
    log.info("Applying logit transform...")
    logits = clip_and_logit(prices)

    # Compute returns
    log.info("Computing returns...")
    returns_raw = compute_returns(logits)

    # Winsorize
    log.info(f"Winsorizing at [{WINSOR_LO}, {WINSOR_HI}]...")
    returns = winsorize(returns_raw, WINSOR_LO, WINSOR_HI)

    # Overlap matrix
    log.info("Computing overlap matrix...")
    overlap = compute_overlap_matrix(returns)
    n_pairs = np.sum(overlap >= MIN_OVERLAP_DAYS)
    n_total = overlap.shape[0] ** 2 - overlap.shape[0]  # exclude diagonal
    log.info(
        f"Pairs with >= {MIN_OVERLAP_DAYS} days overlap: "
        f"{n_pairs} / {n_total} ({100*n_pairs/max(n_total,1):.1f}%)"
    )

    # Liquidity metadata
    log.info("Building liquidity metadata...")
    liquidity = build_liquidity_metadata(eligible, returns)

    # ── Save everything as Parquet ───────────────────────────────────
    ensure_dirs(PROCESSED_DIR)

    try:
        import pandas as pd

        date_strings = [d.isoformat() for d in date_index]

        # Logit prices
        df_logits = pd.DataFrame(logits, index=date_strings, columns=token_ids)
        df_logits.index.name = "date"
        df_logits.to_parquet(LOGIT_PRICES_FILE)
        log.info(f"Saved logit prices: {LOGIT_PRICES_FILE}")

        # Returns
        df_returns = pd.DataFrame(returns, index=date_strings, columns=token_ids)
        df_returns.index.name = "date"
        df_returns.to_parquet(RETURNS_FILE)
        log.info(f"Saved returns: {RETURNS_FILE}")

        # Overlap matrix
        df_overlap = pd.DataFrame(overlap, index=token_ids, columns=token_ids)
        df_overlap.to_parquet(OVERLAP_MATRIX_FILE)
        log.info(f"Saved overlap matrix: {OVERLAP_MATRIX_FILE}")

        # Liquidity metadata
        df_liq = pd.DataFrame(liquidity)
        df_liq.to_parquet(LIQUIDITY_FILE, index=False)
        log.info(f"Saved liquidity metadata: {LIQUIDITY_FILE}")

    except ImportError:
        log.warning("pandas not available — saving as .npy/.json fallback")
        np.save(LOGIT_PRICES_FILE.replace(".parquet", ".npy"), logits)
        np.save(RETURNS_FILE.replace(".parquet", ".npy"), returns)
        np.save(OVERLAP_MATRIX_FILE.replace(".parquet", ".npy"), overlap)
        from utils import save_json
        save_json(liquidity, LIQUIDITY_FILE.replace(".parquet", ".json"))

    # Print summary stats
    log.info("\n── SUMMARY ──")
    log.info(f"  Tokens:             {len(token_ids)}")
    log.info(f"  Date range:         {date_index[0]} → {date_index[-1]}")
    log.info(f"  Price matrix:       {prices.shape}")
    log.info(f"  Non-NaN prices:     {np.sum(~np.isnan(prices)):,}")
    log.info(f"  Non-NaN returns:    {np.sum(~np.isnan(returns)):,}")
    log.info(f"  Eligible pairs:     {n_pairs} (>= {MIN_OVERLAP_DAYS}d overlap)")

    return {
        "logits": logits,
        "returns": returns,
        "overlap": overlap,
        "token_ids": token_ids,
        "date_index": date_index,
    }


if __name__ == "__main__":
    main()
