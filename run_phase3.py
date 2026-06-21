"""
Phase 3 Master Runner
=====================
Runs Phase 3 steps sequentially:
  3.1  Walk-forward forecasting engine (covers roadmap 3.1–3.3)   [implemented]
  3.2  Forecast evaluation (roadmap 3.4)                          [stub]
  3.3  Calibration analysis (roadmap 3.5)                         [stub]
  3.4  Event-conditional network analysis (roadmap 3.6)           [stub]

Usage:
    python run_phase3.py            # all steps
    python run_phase3.py --from 2   # resume from evaluation
    python run_phase3.py --only 1   # only the walk-forward engine
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(__file__))

from utils import get_logger

log = get_logger("phase3")

import step3_1_walkforward
import step3_2_evaluate
import step3_3_calibration
import step3_4_event_analysis

STEPS = {
    1: ("Walk-forward forecasting", step3_1_walkforward.main),
    2: ("Forecast evaluation",      step3_2_evaluate.main),
    3: ("Calibration analysis",     step3_3_calibration.main),
    4: ("Event-conditional",        step3_4_event_analysis.main),
}


def run_step(step_num: int):
    name, func = STEPS[step_num]
    log.info(f"\n{'='*70}")
    log.info(f"PHASE 3 — STEP {step_num}: {name}")
    log.info(f"{'='*70}")
    t0 = time.time()
    try:
        result = func()
        log.info(f"✓ Step {step_num} completed in {time.time()-t0:.1f}s")
        return result
    except Exception as e:
        log.error(f"✗ Step {step_num} FAILED after {time.time()-t0:.1f}s: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Walk-Forward Forecasting & Validation")
    parser.add_argument("--from", dest="from_step", type=int, default=1)
    parser.add_argument("--only", type=int, default=None)
    args = parser.parse_args()

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   PHASE 3: WALK-FORWARD FORECASTING & VALIDATION    ║")
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
            run_step(step_num)

    total = time.time() - t_total
    log.info(f"\n{'='*70}")
    log.info(f"PHASE 3 COMPLETE — Total time: {total:.1f}s ({total/60:.1f}m)")
    log.info(f"{'='*70}")

    log.info("\nOutput files:")
    from config import FORECASTS_FILE, WF_METRICS_FILE, CALIBRATION_FILE, EVENT_ANALYSIS_FILE
    for label, path in [
        ("Forecasts",          FORECASTS_FILE),
        ("Forecast metrics",   WF_METRICS_FILE),
        ("Calibration",        CALIBRATION_FILE),
        ("Event analysis",     EVENT_ANALYSIS_FILE),
    ]:
        exists = "✓" if os.path.exists(path) else "✗"
        log.info(f"  {exists} {label:18s} → {path}")


if __name__ == "__main__":
    main()
