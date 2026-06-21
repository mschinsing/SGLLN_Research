"""
run_all.py — full reproduction driver
=====================================
Runs the entire pipeline (Phase 0 → 5 + reporting) in dependency order, each as
its own subprocess (the documented entry points). Pure orchestration.

Usage:
    python run_all.py                # everything, incl. the ~1h Phase 0 data fetch
    python run_all.py --skip-data    # skip Phase 0 fetch (reuse data/processed/*)
    python run_all.py --skip-trades  # run Phase 0 but skip the slow, unused trades step

Phase 0 (data acquisition) is the long pole (~1h, network-bound). Everything
downstream runs in well under ~15 min. All parameters live in config.py;
RANDOM_SEED=42 fixes stochastic steps.
"""
import sys, os, time, subprocess, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def _run(label, script, args=()):
    print(f"\n{'='*72}\n>>> {label}\n{'='*72}", flush=True)
    t0 = time.time()
    r = subprocess.run([PY, os.path.join(HERE, script), *args], cwd=HERE)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"\n✗ {label} FAILED (exit {r.returncode}) after {dt:.0f}s — stopping.", flush=True)
        sys.exit(r.returncode)
    print(f"✓ {label} done in {dt:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Reproduce the full pipeline")
    ap.add_argument("--skip-data", action="store_true",
                    help="skip Phase 0 fetch (assumes data/processed/* already exists)")
    ap.add_argument("--skip-trades", action="store_true",
                    help="run Phase 0 but skip the slow, downstream-unused trades step")
    args = ap.parse_args()

    t0 = time.time()

    # Phase 0 — data acquisition & preprocessing  (the long pole)
    if not args.skip_data:
        _run("Phase 0 — data acquisition",
             "run_phase0.py", ["--skip-trades"] if args.skip_trades else [])

    # Phases 1–3 — analysis pipeline
    _run("Phase 1 — statistical lead-lag discovery", "run_phase1.py")
    _run("Phase 2 — semantic grounding", "run_phase2.py")
    _run("Phase 3 — walk-forward forecasting & validation", "run_phase3.py")

    # Robustness / synthesis / extensions  (dependency order)
    _run("Common-factor robustness pre-check", "check_common_factor.py")
    _run("Phase 4 — ablations", "phase4_ablations.py")
    _run("Phase 5 — synthesis & hypothesis adjudication", "phase5_synthesis.py")
    _run("Extension — common-factor figures", "common_factor_figures.py")
    _run("Extension — within-politics deep dive", "politics_deepdive.py")

    # Reporting deck last (consumes everything above)
    _run("Presentation figure deck", "make_report_figures.py")

    total = time.time() - t0
    print(f"\n{'='*72}\nALL DONE in {total/60:.1f} min.\n"
          f"  Results : FINDINGS.md, HYPOTHESES.md\n"
          f"  Figures : data/processed/presentation/  (00–12)\n"
          f"  Data    : data/processed/\n{'='*72}", flush=True)


if __name__ == "__main__":
    main()
