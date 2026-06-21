"""
Step 1.5 — Meta-flow graph & cluster ranking
============================================
Aggregates the node-level directed network into a cluster-level "meta-flow" and
ranks clusters by how much they lead the rest of the system:

  F_ab = (1/|C_a||C_b|) Σ_{i∈C_a} Σ_{j∈C_b} (A_ij − A_ji)   net flow a→b (size-normalized)
  L(a) = Σ_{b≠a} F_ab                                        cluster leadingness

F is skew-symmetric (F_ab = −F_ba). A high positive L(a) means cluster a tends to
lead the others (information flows out of it). Renders the directed, edge-weighted
meta-flow graph.

Input:  data/processed/phase1/adjacency_matrix.parquet
        data/processed/phase1/clusters.json
        data/processed/eligible_markets.json   (category mix per cluster, for labels)
Output: data/processed/phase1/meta_flow.parquet     (k×k F matrix)
        data/processed/phase1/cluster_ranking.csv
        data/processed/phase1/figures/meta_flow.png
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
    ADJACENCY_FILE, CLUSTERS_FILE, ELIGIBLE_MARKETS_FILE,
    METAFLOW_FILE, CLUSTER_RANKING_FILE, FIGURES_DIR, PHASE1_DIR,
)
from utils import get_logger, ensure_dirs, load_json

log = get_logger("step1.5_metaflow")


def _meta_flow(adj, labels, k):
    """Size-normalized net inter-cluster flow F (k×k, skew-symmetric)."""
    net = adj - adj.T                          # node-level net flow i→j
    members = [np.nonzero(labels == a)[0] for a in range(k)]
    flow = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            if a == b or len(members[a]) == 0 or len(members[b]) == 0:
                continue
            block = net[np.ix_(members[a], members[b])]
            flow[a, b] = block.sum() / (len(members[a]) * len(members[b]))
    return flow


def _dominant_category(labels, cats, a):
    """Most common platform category among cluster a's members (for a readable label)."""
    members = cats[labels == a]
    if len(members) == 0:
        return "—"
    vals, counts = np.unique(members, return_counts=True)
    return f"{vals[counts.argmax()]} ({counts.max()}/{len(members)})"


def _draw(flow, leadingness, sizes, dom_cats, out_path):
    k = flow.shape[0]
    g = nx.DiGraph()
    for a in range(k):
        g.add_node(a)
    # one directed edge per pair in the net-positive direction
    for a in range(k):
        for b in range(a + 1, k):
            w = flow[a, b]
            if w > 0:
                g.add_edge(a, b, weight=w)
            elif w < 0:
                g.add_edge(b, a, weight=-w)

    pos = nx.circular_layout(g)
    fig, ax = plt.subplots(figsize=(9, 7))
    node_sizes = [300 + 40 * sizes[a] for a in range(k)]
    node_colors = [leadingness[a] for a in range(k)]
    nodes = nx.draw_networkx_nodes(g, pos, node_size=node_sizes, node_color=node_colors,
                                   cmap="coolwarm", ax=ax)
    weights = [g[u][v]["weight"] for u, v in g.edges()]
    wmax = max(weights) if weights else 1.0
    nx.draw_networkx_edges(g, pos, width=[1 + 4 * w / wmax for w in weights],
                           edge_color="#555", arrowsize=18,
                           connectionstyle="arc3,rad=0.08", ax=ax)
    labels_txt = {a: f"C{a}\n{dom_cats[a].split(' (')[0]}\nL={leadingness[a]:+.3f}"
                  for a in range(k)}
    nx.draw_networkx_labels(g, pos, labels=labels_txt, font_size=8, ax=ax)
    fig.colorbar(nodes, ax=ax, label="leadingness L(a)")
    ax.set_title("Meta-flow graph (edge = net inter-cluster lead–lag flow)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    log.info("=" * 60)
    log.info("STEP 1.5: Meta-flow graph & cluster ranking")
    log.info("=" * 60)

    adj_df = pd.read_parquet(ADJACENCY_FILE)
    token_ids = list(adj_df.columns)
    adj = adj_df.to_numpy(dtype=float)

    clusters = load_json(CLUSTERS_FILE)
    k = clusters["k_selected"]
    label_map = clusters["labels"]
    labels = np.array([label_map[t] for t in token_ids])
    log.info(f"Loaded k={k} clustering over {len(token_ids)} nodes")

    meta = {r["token_id"]: r.get("category", "") for r in load_json(ELIGIBLE_MARKETS_FILE)}
    cats = np.array([meta.get(t, "") for t in token_ids])

    F = _meta_flow(adj, labels, k)
    leadingness = F.sum(axis=1)                 # L(a) = Σ_b F_ab
    sizes = np.bincount(labels, minlength=k)
    dom_cats = [_dominant_category(labels, cats, a) for a in range(k)]

    ensure_dirs([PHASE1_DIR, FIGURES_DIR])
    pd.DataFrame(F, index=[f"C{a}" for a in range(k)],
                 columns=[f"C{a}" for a in range(k)]).to_parquet(METAFLOW_FILE)

    rank = pd.DataFrame({
        "cluster": [f"C{a}" for a in range(k)],
        "size": sizes,
        "leadingness": leadingness,
        "dominant_category": dom_cats,
    }).sort_values("leadingness", ascending=False).reset_index(drop=True)
    rank.to_csv(CLUSTER_RANKING_FILE, index=False)

    fig_path = os.path.join(FIGURES_DIR, "meta_flow.png")
    _draw(F, leadingness, sizes, dom_cats, fig_path)

    log.info(f"Saved meta-flow matrix: {METAFLOW_FILE}")
    log.info(f"Saved cluster ranking:  {CLUSTER_RANKING_FILE}")
    log.info(f"Saved figure:           {fig_path}")
    log.info("\n── CLUSTER RANKING (most → least leading) ──")
    for _, r in rank.iterrows():
        log.info(f"  {r['cluster']}  L={r['leadingness']:+.4f}  "
                 f"n={int(r['size']):3d}  {r['dominant_category']}")

    return {"F": F, "leadingness": leadingness, "ranking": rank}


if __name__ == "__main__":
    main()
