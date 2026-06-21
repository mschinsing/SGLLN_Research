"""
Step 2.4 — Semantic-statistical fusion
======================================
Fuses the Phase 1 spectral node embeddings v_i ∈ R^{2k} with the semantic
embeddings ê_i ∈ R^384, then re-clusters:

    z_i = [ v_block_i | α · e_block_i ]

Each block is first per-feature standardized AND scaled by 1/√(dim): after
standardization a d-dim block carries total variance d, so the 384-dim semantic
block would otherwise dominate the 14-dim spectral block ~27× in Euclidean
distance. The 1/√(dim) scaling gives each block unit total variance, so **α is an
interpretable equal-weight knob** — α=1 weights the two signals equally, α<1 is
statistical-dominated, α>1 semantic-dominated.

k-means on z → fused clusters. The sweep shows how the fused partition
interpolates between the pure statistical and pure semantic clusters (ARI/NMI).

Input:  data/processed/phase1/node_embeddings.npy   (v_i, canonical order)
        data/processed/phase2/contract_embeddings.npy (ê_i, canonical order)
        data/processed/phase1/clusters.json, phase2/semantic_clusters.json
        data/processed/phase1/adjacency_matrix.parquet (canonical order)
Output: data/processed/phase2/fused_clusters.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from config import (
    NODE_EMBEDDINGS_FILE, SEM_EMBEDDINGS_FILE, ADJACENCY_FILE,
    CLUSTERS_FILE, SEM_CLUSTERS_FILE, FUSED_CLUSTERS_FILE, PHASE2_DIR,
    FUSION_ALPHAS, RANDOM_SEED,
)
from utils import get_logger, ensure_dirs, save_json, load_json

log = get_logger("step2.4_fusion")

PRIMARY_ALPHA = 1.0    # alpha whose labels are saved as the canonical fused partition


def _standardize(x):
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def main():
    log.info("=" * 60)
    log.info("STEP 2.4: Semantic-statistical fusion")
    log.info("=" * 60)

    token_ids = list(pd.read_parquet(ADJACENCY_FILE).columns)
    v = np.load(NODE_EMBEDDINGS_FILE)          # (n, 2k), canonical order
    e = np.load(SEM_EMBEDDINGS_FILE)           # (n, 384), canonical order
    log.info(f"v_i={v.shape}  ê_i={e.shape}")

    stat_map = load_json(CLUSTERS_FILE)
    k = int(stat_map["k_selected"])
    statistical = np.array([stat_map["labels"][t] for t in token_ids])
    sem_map = load_json(SEM_CLUSTERS_FILE)["labels"]
    semantic = np.array([sem_map[t] for t in token_ids])

    # Standardize per feature, then scale each block by 1/sqrt(dim) so both blocks
    # have unit TOTAL variance and α is an interpretable equal-weight knob.
    v_block = _standardize(v) / np.sqrt(v.shape[1])
    e_block = _standardize(e) / np.sqrt(e.shape[1])

    log.info(f"\nFusion sweep (k={k}, blocks scaled by 1/√dim) — ARI/NMI vs pure partitions:")
    sweep = []
    primary_labels = None
    for alpha in sorted(set(FUSION_ALPHAS) | {PRIMARY_ALPHA}):
        z = np.concatenate([v_block, alpha * e_block], axis=1)
        labels = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit_predict(z)
        ari_s = adjusted_rand_score(labels, statistical)
        nmi_s = normalized_mutual_info_score(labels, statistical)
        ari_e = adjusted_rand_score(labels, semantic)
        nmi_e = normalized_mutual_info_score(labels, semantic)
        flag = "  ← saved" if alpha == PRIMARY_ALPHA else ""
        log.info(f"  α={alpha:<4}: vs statistical ARI={ari_s:+.3f} NMI={nmi_s:.3f} | "
                 f"vs semantic ARI={ari_e:+.3f} NMI={nmi_e:.3f}{flag}")
        sweep.append({"alpha": alpha, "ari_statistical": float(ari_s),
                      "nmi_statistical": float(nmi_s), "ari_semantic": float(ari_e),
                      "nmi_semantic": float(nmi_e),
                      "sizes": np.bincount(labels, minlength=k).tolist()})
        if alpha == PRIMARY_ALPHA:
            primary_labels = labels

    ensure_dirs(PHASE2_DIR)
    save_json({
        "k": k,
        "primary_alpha": PRIMARY_ALPHA,
        "sweep": sweep,
        "labels": {t: int(c) for t, c in zip(token_ids, primary_labels)},
    }, FUSED_CLUSTERS_FILE)
    log.info(f"\nSaved fused clusters (α={PRIMARY_ALPHA}): {FUSED_CLUSTERS_FILE}")

    return {"sweep": sweep, "labels": primary_labels}


if __name__ == "__main__":
    main()
