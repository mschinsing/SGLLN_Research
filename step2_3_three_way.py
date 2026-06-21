"""
Step 2.3 — Three-way cluster comparison
=======================================
Compares three partitions of the same contracts:
  1. statistical clusters  (Phase 1, clusters.json k_selected)
  2. semantic clusters     (Phase 2.2)
  3. platform categories   (Polymarket metadata)

Metrics:
  * ARI + NMI between every pair of partitions.
  * Semantic Coherence Score per statistical cluster:
        SCS(C_a) = mean_{i<j∈C_a} W^sem_ij
    with z-scores vs a random-clustering null (size-preserving label shuffles).

A high SCS z-score means a statistically-discovered cluster is *also* internally
coherent in meaning — i.e. the lead–lag structure is semantically grounded.

Input:  data/processed/phase1/clusters.json
        data/processed/phase2/semantic_clusters.json
        data/processed/phase2/semantic_similarity.parquet
        data/processed/eligible_markets.json
Output: data/processed/phase2/three_way_comparison.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from config import (
    CLUSTERS_FILE, SEM_CLUSTERS_FILE, SEM_SIM_FILE, ELIGIBLE_MARKETS_FILE,
    THREEWAY_FILE, PHASE2_DIR, SCS_NULL_B, RANDOM_SEED,
)
from utils import get_logger, ensure_dirs, save_json, load_json

log = get_logger("step2.3_three_way")


def _scs(labels, sim, a):
    """Mean within-cluster semantic similarity for cluster `a`."""
    members = np.nonzero(labels == a)[0]
    if len(members) < 2:
        return np.nan
    sub = sim[np.ix_(members, members)]
    iu = np.triu_indices(len(members), k=1)
    return float(sub[iu].mean())


def _scs_zscores(labels, sim, k, n_reps, seed):
    """Z-score each cluster's SCS against size-preserving random relabelings."""
    rng = np.random.default_rng(seed)
    observed = np.array([_scs(labels, sim, a) for a in range(k)])
    null = np.full((n_reps, k), np.nan)
    for b in range(n_reps):
        perm = rng.permutation(labels)
        for a in range(k):
            null[b, a] = _scs(perm, sim, a)
    mean = np.nanmean(null, axis=0)
    std = np.nanstd(null, axis=0)
    z = np.where(std > 0, (observed - mean) / std, np.nan)
    return observed, mean, z


def main():
    log.info("=" * 60)
    log.info("STEP 2.3: Three-way cluster comparison")
    log.info("=" * 60)

    sim_df = pd.read_parquet(SEM_SIM_FILE)
    token_ids = list(sim_df.columns)
    sim = sim_df.to_numpy(dtype=float)

    stat_map = load_json(CLUSTERS_FILE)["labels"]
    sem_map = load_json(SEM_CLUSTERS_FILE)["labels"]
    cat_map = {r["token_id"]: r.get("category", "") for r in load_json(ELIGIBLE_MARKETS_FILE)}

    statistical = np.array([stat_map[t] for t in token_ids])
    semantic = np.array([sem_map[t] for t in token_ids])
    _, category = np.unique([cat_map.get(t, "") for t in token_ids], return_inverse=True)
    k = int(statistical.max()) + 1

    # ── Pairwise ARI / NMI ───────────────────────────────────────────────
    pairs = {
        "statistical_vs_semantic": (statistical, semantic),
        "statistical_vs_category": (statistical, category),
        "semantic_vs_category":    (semantic, category),
    }
    comparison = {}
    log.info("\n── Pairwise agreement ──")
    for name, (x, y) in pairs.items():
        ari = float(adjusted_rand_score(x, y))
        nmi = float(normalized_mutual_info_score(x, y))
        comparison[name] = {"ARI": ari, "NMI": nmi}
        log.info(f"  {name:28s}  ARI={ari:+.3f}  NMI={nmi:.3f}")

    # ── Semantic Coherence Score of statistical clusters ─────────────────
    observed, null_mean, z = _scs_zscores(statistical, sim, k, SCS_NULL_B, RANDOM_SEED)
    global_mean = float(sim[np.triu_indices(len(token_ids), k=1)].mean())
    log.info(f"\n── Semantic Coherence (statistical clusters) vs null "
             f"(global mean W^sem={global_mean:.3f}) ──")
    scs_rows = []
    for a in range(k):
        log.info(f"  C{a}: SCS={observed[a]:.3f}  null={null_mean[a]:.3f}  z={z[a]:+.2f}")
        scs_rows.append({"cluster": a, "scs": float(observed[a]),
                         "null_mean": float(null_mean[a]), "z": float(z[a])})
    n_sig = int(np.sum(z > 1.96))
    log.info(f"  Clusters with SCS z>1.96 (semantically coherent): {n_sig}/{k}")

    ensure_dirs(PHASE2_DIR)
    save_json({
        "pairwise": comparison,
        "global_mean_wsem": global_mean,
        "semantic_coherence": scs_rows,
        "n_coherent_clusters": n_sig,
        "k": k,
    }, THREEWAY_FILE)
    log.info(f"\nSaved three-way comparison: {THREEWAY_FILE}")

    return {"comparison": comparison, "scs": observed, "scs_z": z}


if __name__ == "__main__":
    main()
