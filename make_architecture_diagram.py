"""
make_architecture_diagram.py — generate the system architecture diagram
=======================================================================
Draws the Phase 0→5 data-flow (modules + artifacts) for the report/slides.

  python make_architecture_diagram.py  ->  data/processed/presentation/00_architecture.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from config import PRESENTATION_DIR
from utils import get_logger, ensure_dirs

log = get_logger("architecture_diagram")

# (label, detail, color)
BANDS = [
    ("DATA SOURCES", "Polymarket Gamma API (market metadata)  ·  CLOB API (daily price history)", "#718096"),
    ("PHASE 0 — Data acquisition & preprocessing",
     "fetch → 5-criterion eligibility funnel → 354 contracts, 8 categories → logit returns · overlap matrix · event calendar", "#2b6cb0"),
    ("PHASE 1 — Statistical lead-lag discovery",
     "dcor CCF-AUC (62,481 pairs) → bootstrap τ_dep floor → directed adjacency → Hermitian spectral clustering (k=7) → meta-flow + leadingness → liquidity controls", "#2c7a7b"),
    ("PHASE 2 — Semantic grounding",
     "Sentence-BERT embeddings (384-d) → semantic spectral clustering → three-way comparison (ARI/NMI) → semantic-statistical fusion → SPS edge filter", "#6b46c1"),
    ("PHASE 3 — Walk-forward forecasting & validation",
     "strictly-causal walk-forward → 3 signal variants → evaluation (directional acc · FDR · bootstrap CIs · break-even cost) → calibration → event-conditional |λ1|", "#b7791f"),
    ("PHASE 4 / 5 — Ablations & synthesis",
     "robustness ablations (metric · clustering method · common factor)   |   hypothesis adjudication H1–H4", "#9b2c2c"),
    ("OUTPUTS", "FINDINGS.md  ·  HYPOTHESES.md  ·  presentation figure deck (00–12)  ·  reproducible JSON/parquet artifacts", "#2d3748"),
]

ARTIFACTS = ["markets / prices", "returns.parquet · overlap · embeddings",
             "adjacency · clusters · meta-flow", "semantic clusters · three-way",
             "forecasts · metrics · calibration", "results tables"]


def main():
    ensure_dirs(PRESENTATION_DIR)
    n = len(BANDS)
    fig, ax = plt.subplots(figsize=(12, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n * 2)
    ax.axis("off")

    box_h, gap = 1.4, 0.6
    centers = []
    for i, (label, detail, color) in enumerate(BANDS):
        y = (n - 1 - i) * 2 + 0.3
        cy = y + box_h / 2
        centers.append(cy)
        ax.add_patch(FancyBboxPatch((0.4, y), 9.2, box_h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    facecolor=color, edgecolor="black", alpha=0.92, linewidth=1))
        ax.text(5.0, cy + 0.34, label, ha="center", va="center", fontsize=12.5,
                fontweight="bold", color="white")
        ax.text(5.0, cy - 0.30, detail, ha="center", va="center", fontsize=8.3,
                color="white", wrap=True)

    # downward artifact arrows between bands
    for i in range(n - 1):
        y_top = centers[i] - box_h / 2
        y_bot = centers[i + 1] + box_h / 2
        ax.add_patch(FancyArrowPatch((5.0, y_top), (5.0, y_bot), arrowstyle="-|>",
                                     mutation_scale=18, color="#1a202c", linewidth=1.6))
        ax.text(5.25, (y_top + y_bot) / 2, ARTIFACTS[i], ha="left", va="center",
                fontsize=7.5, style="italic", color="#1a202c")

    ax.text(5.0, n * 2 - 0.15, "Semantically-Grounded Lead-Lag Networks — System Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PRESENTATION_DIR, "00_architecture.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved architecture diagram: {path}")


if __name__ == "__main__":
    main()
