"""
Step 2.2 — Semantic clustering
==============================
Spectral clustering on the semantic similarity matrix W^sem (symmetric,
non-negative) → semantic clusters derived purely from text, with no price data.

Cluster count matches the Phase 1 statistical partition (clusters.json
k_selected) so the two partitions are directly comparable in step 2.3.

Input:  data/processed/phase2/semantic_similarity.parquet
        data/processed/phase1/clusters.json   (for k_selected)
Output: data/processed/phase2/semantic_clusters.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering

from config import SEM_SIM_FILE, CLUSTERS_FILE, SEM_CLUSTERS_FILE, PHASE2_DIR, RANDOM_SEED
from utils import get_logger, ensure_dirs, save_json, load_json

log = get_logger("step2.2_semantic_clustering")


def main():
    log.info("=" * 60)
    log.info("STEP 2.2: Semantic spectral clustering (text only)")
    log.info("=" * 60)

    sim_df = pd.read_parquet(SEM_SIM_FILE)
    token_ids = list(sim_df.columns)
    sim = sim_df.to_numpy(dtype=float)

    k = int(load_json(CLUSTERS_FILE)["k_selected"])
    log.info(f"Clustering {len(token_ids)} contracts into k={k} (matched to statistical k_selected)")

    sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                            assign_labels="kmeans", random_state=RANDOM_SEED)
    labels = sc.fit_predict(sim)

    ensure_dirs(PHASE2_DIR)
    save_json({
        "k": k,
        "labels": {t: int(c) for t, c in zip(token_ids, labels)},
        "sizes": np.bincount(labels, minlength=k).tolist(),
    }, SEM_CLUSTERS_FILE)
    log.info(f"Saved semantic clusters: {SEM_CLUSTERS_FILE}")
    log.info(f"Cluster sizes: {np.bincount(labels, minlength=k).tolist()}")

    return {"labels": labels, "token_ids": token_ids, "k": k}


if __name__ == "__main__":
    main()
