"""
Step 1.3 — Directed adjacency matrix construction
=================================================
Builds the dependence-gated directed adjacency matrix from the dcor lead–lag
integrals (step 1.1) and the global dependence floor τ_dep (step 1.2):

    S_ij = sign(I_ij - I_ji) · max(I_ij, I_ji) / (I_ij + I_ji)     (CCF-AUC)
    A_ij = [S_ij]_+ · max(I_ij, I_ji) · 1[ max(I_ij, I_ji) > τ_dep ]

  * [S_ij]_+ = max(S_ij, 0) keeps only the leading direction of each pair
    (S is sign-antisymmetric, so exactly one of A_ij / A_ji is non-zero).
  * The indicator gates out pairs whose absolute dependence is below τ_dep.

Vectorized over the dense I matrix: with I[i,j] = I(i→j),
    M = max(I, Iᵀ) elementwise,  denom = I + Iᵀ,  S = sign(I−Iᵀ)·M/denom.

Input:  data/processed/phase1/idcor_matrix.parquet   (dense directed I_dcor)
        data/processed/phase1/tau_dep.json
Output: data/processed/phase1/adjacency_matrix.parquet  (354×354 directed, weighted)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from config import IDCOR_MATRIX_FILE, TAU_DEP_FILE, ADJACENCY_FILE, PHASE1_DIR
from utils import get_logger, ensure_dirs, load_json

log = get_logger("step1.3_adjacency")


def main():
    log.info("=" * 60)
    log.info("STEP 1.3: Directed adjacency matrix construction")
    log.info("=" * 60)

    idcor_df = pd.read_parquet(IDCOR_MATRIX_FILE)
    token_ids = list(idcor_df.columns)
    I = idcor_df.to_numpy(dtype=float)              # I[i, j] = I(i -> j)
    n = I.shape[0]

    tau_dep = float(load_json(TAU_DEP_FILE)["tau_dep"])
    log.info(f"Loaded I matrix {I.shape} and τ_dep = {tau_dep:.4f}")

    i_t = I.T
    M = np.maximum(I, i_t)                           # max(I_ij, I_ji), symmetric
    denom = I + i_t
    # S = CCF-AUC, sign-antisymmetric; 0 where there is no dependence at all.
    ratio = np.divide(M, denom, out=np.zeros_like(M), where=denom > 0)
    S = np.sign(I - i_t) * ratio

    A = np.maximum(S, 0.0) * M                       # keep leading direction, weight by max-I
    A[M <= tau_dep] = 0.0                            # dependence gate
    np.fill_diagonal(A, 0.0)

    ensure_dirs(PHASE1_DIR)
    pd.DataFrame(A, index=token_ids, columns=token_ids).to_parquet(ADJACENCY_FILE)
    log.info(f"Saved adjacency: {ADJACENCY_FILE}")

    # Diagnostics
    n_edges = int((A > 0).sum())
    possible = n * (n - 1)
    out_deg = (A > 0).sum(axis=1)
    in_deg = (A > 0).sum(axis=0)
    nz = A[A > 0]
    log.info("\n── SUMMARY ──")
    log.info(f"  Nodes:               {n}")
    log.info(f"  Directed edges:      {n_edges:,} / {possible:,} ({100*n_edges/possible:.2f}% density)")
    log.info(f"  Edge weight mean/med/max: {nz.mean():.4f} / {np.median(nz):.4f} / {nz.max():.4f}")
    log.info(f"  Out-degree max/mean: {int(out_deg.max())} / {out_deg.mean():.1f}")
    log.info(f"  In-degree  max/mean: {int(in_deg.max())} / {in_deg.mean():.1f}")
    log.info(f"  Isolated nodes (deg 0): {int(((out_deg + in_deg) == 0).sum())}")

    return {"adjacency": A, "token_ids": token_ids, "n_edges": n_edges, "tau_dep": tau_dep}


if __name__ == "__main__":
    main()
