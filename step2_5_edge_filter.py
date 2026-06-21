"""
Step 2.5 — Semantic edge filtering at the cluster level
=======================================================
Scores each meta-flow edge by the average semantic similarity between the two
clusters it connects, then prunes edges that are semantically implausible:

    SPS(a, b) = (1/|C_a||C_b|) Σ_{i∈C_a} Σ_{j∈C_b} W^sem_ij

Edges of the directed meta-flow graph (step 1.5) are retained only where
SPS(a,b) exceeds the SPS_PERCENTILE threshold → a semantically-filtered network
in which surviving lead–lag links also make topical sense.

Input:  data/processed/phase1/meta_flow.parquet
        data/processed/phase1/clusters.json
        data/processed/phase2/semantic_similarity.parquet
Output: data/processed/phase2/semantic_filtered_metaflow.parquet
        data/processed/phase2/figures/semantic_filtered_metaflow.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from config import (
    METAFLOW_FILE, CLUSTERS_FILE, SEM_SIM_FILE,
    SEM_FILTERED_METAFLOW_FILE, PHASE2_DIR, PHASE2_FIGURES_DIR, SPS_PERCENTILE,
)
from utils import get_logger, ensure_dirs, load_json

log = get_logger("step2.5_edge_filter")


def _sps_matrix(sim, labels, k):
    members = [np.nonzero(labels == a)[0] for a in range(k)]
    sps = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            if len(members[a]) and len(members[b]):
                sps[a, b] = sim[np.ix_(members[a], members[b])].mean()
    return sps


def _draw(flow, sps, retained, out_path):
    k = flow.shape[0]
    g = nx.DiGraph()
    g.add_nodes_from(range(k))
    for a in range(k):
        for b in range(a + 1, k):
            if not retained[a, b]:
                continue
            w = flow[a, b]
            if w > 0:
                g.add_edge(a, b, weight=w)
            elif w < 0:
                g.add_edge(b, a, weight=-w)

    pos = nx.circular_layout(nx.complete_graph(k))
    fig, ax = plt.subplots(figsize=(8, 7))
    nx.draw_networkx_nodes(g, pos, node_size=700, node_color="#cbd5e0", ax=ax)
    nx.draw_networkx_labels(g, pos, {a: f"C{a}" for a in range(k)}, ax=ax)
    if g.edges():
        weights = [g[u][v]["weight"] for u, v in g.edges()]
        wmax = max(weights)
        nx.draw_networkx_edges(g, pos, width=[1 + 4 * w / wmax for w in weights],
                               edge_color="#2b6cb0", arrowsize=18,
                               connectionstyle="arc3,rad=0.08", ax=ax)
    ax.set_title(f"Semantically-filtered meta-flow "
                 f"(SPS > p{SPS_PERCENTILE}; {g.number_of_edges()} edges retained)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    log.info("=" * 60)
    log.info("STEP 2.5: Semantic edge filtering (SPS)")
    log.info("=" * 60)

    flow_df = pd.read_parquet(METAFLOW_FILE)
    flow = flow_df.to_numpy(dtype=float)
    k = flow.shape[0]

    clusters = load_json(CLUSTERS_FILE)
    token_ids = list(pd.read_parquet(SEM_SIM_FILE).columns)
    sim = pd.read_parquet(SEM_SIM_FILE).to_numpy(dtype=float)
    labels = np.array([clusters["labels"][t] for t in token_ids])

    sps = _sps_matrix(sim, labels, k)
    iu = np.triu_indices(k, k=1)
    threshold = float(np.percentile(sps[iu], SPS_PERCENTILE))
    retained = sps > threshold
    np.fill_diagonal(retained, False)

    # Filtered directed meta-flow: zero out semantically-implausible edges
    flow_filt = np.where(retained, flow, 0.0)

    n_edges_before = int(np.sum(np.abs(flow[iu]) > 0))
    n_edges_after = int(np.sum(retained[iu]))
    log.info(f"SPS threshold (p{SPS_PERCENTILE} of inter-cluster SPS): {threshold:.4f}")
    log.info(f"Cluster pairs retained: {n_edges_after}/{len(iu[0])} "
             f"(meta-flow edges {n_edges_before} → {n_edges_after})")

    ensure_dirs([PHASE2_DIR, PHASE2_FIGURES_DIR])
    pd.DataFrame(flow_filt, index=flow_df.index, columns=flow_df.columns).to_parquet(
        SEM_FILTERED_METAFLOW_FILE)
    fig_path = os.path.join(PHASE2_FIGURES_DIR, "semantic_filtered_metaflow.png")
    _draw(flow, sps, retained, fig_path)
    log.info(f"Saved filtered meta-flow: {SEM_FILTERED_METAFLOW_FILE}")
    log.info(f"Saved figure: {fig_path}")

    log.info("\n── Retained inter-cluster links (by |net flow|) ──")
    rows = []
    for a in range(k):
        for b in range(a + 1, k):
            if retained[a, b]:
                lead, lag = (a, b) if flow[a, b] > 0 else (b, a)
                rows.append((abs(flow[a, b]), lead, lag, sps[a, b]))
    for w, lead, lag, s in sorted(rows, reverse=True):
        log.info(f"  C{lead} → C{lag}   net_flow={w:.4f}  SPS={s:.3f}")

    return {"sps": sps, "threshold": threshold, "flow_filtered": flow_filt}


if __name__ == "__main__":
    main()
