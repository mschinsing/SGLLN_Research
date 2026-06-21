"""
Phase 1 Master Runner
=====================
Runs all Phase 1 steps sequentially:
  1.1  Pairwise lead–lag metrics (dcor CCF-AUC + baselines)   [implemented]
  1.2  Significance testing → global dependence floor τ_dep   [stub]
  1.3  Directed adjacency matrix construction                 [stub]
  1.4  Hermitian random-walk spectral clustering              [stub]
  1.5  Meta-flow graph & cluster ranking                      [stub]
  1.6  Liquidity controls (confound check)                    [stub]

Usage:
    python run_phase1.py              # run all steps
    python run_phase1.py --from 3     # resume from step 1.3
    python run_phase1.py --only 1     # run only step 1.1
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(__file__))

from utils import get_logger

log = get_logger("phase1")

import step1_1_leadlag
import step1_2_significance
import step1_3_adjacency
import step1_4_spectral
import step1_5_metaflow
import step1_6_liquidity

STEPS = {
    1: ("Pairwise lead–lag metrics", step1_1_leadlag.main),
    2: ("Significance / tau_dep",    step1_2_significance.main),
    3: ("Directed adjacency",        step1_3_adjacency.main),
    4: ("Spectral clustering",       step1_4_spectral.main),
    5: ("Meta-flow & ranking",       step1_5_metaflow.main),
    6: ("Liquidity controls",        step1_6_liquidity.main),
}


def run_step(step_num: int):
    name, func = STEPS[step_num]
    log.info(f"\n{'='*70}")
    log.info(f"PHASE 1 — STEP {step_num}: {name}")
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
    parser = argparse.ArgumentParser(description="Phase 1: Statistical Lead–Lag Discovery")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from this step (default: 1)")
    parser.add_argument("--only", type=int, default=None,
                        help="Run only this step")
    args = parser.parse_args()

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   PHASE 1: STATISTICAL LEAD–LAG DISCOVERY           ║")
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

    total_elapsed = time.time() - t_total
    log.info(f"\n{'='*70}")
    log.info(f"PHASE 1 COMPLETE — Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    log.info(f"{'='*70}")

    # Output summary
    log.info("\nOutput files:")
    from config import (
        LEADLAG_SCORES_FILE, IDCOR_MATRIX_FILE, TAU_DEP_FILE, ADJACENCY_FILE,
        CLUSTERS_FILE, METAFLOW_FILE, CLUSTER_RANKING_FILE, LIQUIDITY_CONTROLS_FILE,
    )
    for label, path in [
        ("Lead–lag scores",     LEADLAG_SCORES_FILE),
        ("I_dcor matrix",       IDCOR_MATRIX_FILE),
        ("tau_dep",             TAU_DEP_FILE),
        ("Adjacency",           ADJACENCY_FILE),
        ("Clusters",            CLUSTERS_FILE),
        ("Meta-flow",           METAFLOW_FILE),
        ("Cluster ranking",     CLUSTER_RANKING_FILE),
        ("Liquidity controls",  LIQUIDITY_CONTROLS_FILE),
    ]:
        exists = "✓" if os.path.exists(path) else "✗"
        log.info(f"  {exists} {label:22s} → {path}")


if __name__ == "__main__":
    main()
