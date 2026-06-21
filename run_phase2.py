"""
Phase 2 Master Runner
=====================
Runs all Phase 2 steps sequentially:
  2.1  Contract semantic embedding (category-free) + W^sem
  2.2  Semantic spectral clustering
  2.3  Three-way cluster comparison (statistical / semantic / category)
  2.4  Semantic-statistical fusion
  2.5  Semantic edge filtering of the meta-flow graph

Usage:
    python run_phase2.py            # all steps
    python run_phase2.py --from 3   # resume from step 2.3
    python run_phase2.py --only 1   # only step 2.1
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(__file__))

from utils import get_logger

log = get_logger("phase2")

import step2_1_semantic_embedding
import step2_2_semantic_clustering
import step2_3_three_way
import step2_4_fusion
import step2_5_edge_filter

STEPS = {
    1: ("Semantic embedding",      step2_1_semantic_embedding.main),
    2: ("Semantic clustering",     step2_2_semantic_clustering.main),
    3: ("Three-way comparison",    step2_3_three_way.main),
    4: ("Semantic-stat fusion",    step2_4_fusion.main),
    5: ("Semantic edge filtering", step2_5_edge_filter.main),
}


def run_step(step_num: int):
    name, func = STEPS[step_num]
    log.info(f"\n{'='*70}")
    log.info(f"PHASE 2 — STEP {step_num}: {name}")
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
    parser = argparse.ArgumentParser(description="Phase 2: Semantic Grounding")
    parser.add_argument("--from", dest="from_step", type=int, default=1)
    parser.add_argument("--only", type=int, default=None)
    args = parser.parse_args()

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   PHASE 2: SEMANTIC GROUNDING                       ║")
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
    log.info(f"PHASE 2 COMPLETE — Total time: {total:.1f}s ({total/60:.1f}m)")
    log.info(f"{'='*70}")

    log.info("\nOutput files:")
    from config import (
        SEM_EMBEDDINGS_FILE, SEM_SIM_FILE, SEM_CLUSTERS_FILE,
        THREEWAY_FILE, FUSED_CLUSTERS_FILE, SEM_FILTERED_METAFLOW_FILE,
    )
    for label, path in [
        ("Semantic embeddings",   SEM_EMBEDDINGS_FILE),
        ("Semantic similarity",   SEM_SIM_FILE),
        ("Semantic clusters",     SEM_CLUSTERS_FILE),
        ("Three-way comparison",  THREEWAY_FILE),
        ("Fused clusters",        FUSED_CLUSTERS_FILE),
        ("Filtered meta-flow",    SEM_FILTERED_METAFLOW_FILE),
    ]:
        exists = "✓" if os.path.exists(path) else "✗"
        log.info(f"  {exists} {label:22s} → {path}")


if __name__ == "__main__":
    main()
