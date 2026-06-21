"""
Phase 0 Master Runner
======================
Runs all Phase 0 steps sequentially:
  0.1  Fetch market metadata (Gamma API)
  0.2  Fetch price histories (CLOB API)
  0.3  Fetch trade data for liquidity (Data API)
  0.4  Apply filtering funnel
  0.5  Preprocess: logit, returns, winsorize, overlap
  0.6  Supplementary: event calendar & embeddings

Usage:
    python run_phase0.py              # run all steps
    python run_phase0.py --from 4     # resume from step 4
    python run_phase0.py --only 2     # run only step 2
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(__file__))

from utils import get_logger

log = get_logger("phase0")

# Import all step modules
import step0_1_fetch_markets
import step0_2_fetch_prices
import step0_3_fetch_trades
import step0_4_filter_funnel
import step0_5_preprocess
import step0_6_supplementary

STEPS = {
    1: ("Fetch market metadata",   step0_1_fetch_markets.main),
    2: ("Fetch price histories",   step0_2_fetch_prices.main),
    3: ("Fetch trade/liquidity",   step0_3_fetch_trades.main),
    4: ("Filtering funnel",        step0_4_filter_funnel.main),
    5: ("Preprocessing",           step0_5_preprocess.main),
    6: ("Supplementary data",      step0_6_supplementary.main),
}


def run_step(step_num: int):
    name, func = STEPS[step_num]
    log.info(f"\n{'='*70}")
    log.info(f"PHASE 0 — STEP {step_num}: {name}")
    log.info(f"{'='*70}")
    t0 = time.time()
    try:
        result = func()
        elapsed = time.time() - t0
        log.info(f"✓ Step {step_num} completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = time.time() - t0
        log.error(f"✗ Step {step_num} FAILED after {elapsed:.1f}s: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Data Acquisition & Preprocessing")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from this step (default: 1)")
    parser.add_argument("--only", type=int, default=None,
                        help="Run only this step")
    parser.add_argument("--skip-trades", action="store_true",
                        help="Skip step 3 (trade fetching — slowest step)")
    args = parser.parse_args()

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   PHASE 0: DATA ACQUISITION & PREPROCESSING        ║")
    log.info("║   Semantically-Grounded Lead-Lag Networks           ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    t_total = time.time()

    if args.only:
        if args.only not in STEPS:
            log.error(f"Invalid step: {args.only}. Valid: {list(STEPS.keys())}")
            sys.exit(1)
        run_step(args.only)
    else:
        for step_num in sorted(STEPS.keys()):
            if step_num < args.from_step:
                log.info(f"Skipping step {step_num} (--from {args.from_step})")
                continue
            if args.skip_trades and step_num == 3:
                log.info("Skipping step 3 (--skip-trades)")
                continue
            run_step(step_num)

    total_elapsed = time.time() - t_total
    log.info(f"\n{'='*70}")
    log.info(f"PHASE 0 COMPLETE — Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    log.info(f"{'='*70}")

    # Print output summary
    log.info("\nOutput files:")
    from config import (
        MARKETS_RAW_FILE, PRICES_RAW_DIR, TRADES_RAW_DIR,
        FILTER_FUNNEL_FILE, ELIGIBLE_MARKETS_FILE,
        LOGIT_PRICES_FILE, RETURNS_FILE, OVERLAP_MATRIX_FILE,
        LIQUIDITY_FILE, EVENTS_FILE, EMBEDDINGS_DIR,
    )
    for label, path in [
        ("Markets (raw)",       MARKETS_RAW_FILE),
        ("Prices (raw)",        PRICES_RAW_DIR),
        ("Trades (raw)",        TRADES_RAW_DIR),
        ("Filter funnel",       FILTER_FUNNEL_FILE),
        ("Eligible markets",    ELIGIBLE_MARKETS_FILE),
        ("Logit prices",        LOGIT_PRICES_FILE),
        ("Returns",             RETURNS_FILE),
        ("Overlap matrix",      OVERLAP_MATRIX_FILE),
        ("Liquidity metadata",  LIQUIDITY_FILE),
        ("Event calendar",      EVENTS_FILE),
        ("Embeddings",          EMBEDDINGS_DIR),
    ]:
        exists = "✓" if os.path.exists(path) else "✗"
        log.info(f"  {exists} {label:25s} → {path}")


if __name__ == "__main__":
    main()
