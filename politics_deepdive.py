"""
politics_deepdive.py — within-politics sub-topic lead-lag
========================================================
The lead-lag structure is predominantly *within* Politics (196 contracts). This
asks the novel follow-up: which political sub-topics lead which? Pure reuse —
filter the existing directed adjacency to political contracts and compute the
sub-topic meta-flow (does the presidential race lead state-level markets? do
nomination contracts lead the general election?).

  python politics_deepdive.py

Output: data/processed/phase5/politics_subflow.json
        data/processed/presentation/12_politics_subflow.png
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
    ADJACENCY_FILE, ELIGIBLE_MARKETS_FILE, POLITICS_SUBFLOW_FILE,
    PHASE5_DIR, PRESENTATION_DIR, RANDOM_SEED,
)
from utils import get_logger, ensure_dirs, save_json, load_json

log = get_logger("politics_deepdive")
_RNG = np.random.default_rng(RANDOM_SEED)
PERM = 2000

SWING = ["pennsylvania", "michigan", "georgia", "arizona", "wisconsin", "nevada",
         "north carolina", "minnesota", "virginia", "florida", "ohio", "new hampshire"]


def _subtopic(q):
    ql = q.lower()
    if any(s in ql for s in SWING) or "swing" in ql:
        return "state-level"
    if any(k in ql for k in ["nominee", "nomination", "drop out", "withdraw", "resign",
                             "primary", "dnc", "rnc", "replace"]):
        return "nomination"
    if any(k in ql for k in ["senate", "house", "congress", "governor", "seat", "representative"]):
        return "congress"
    if any(k in ql for k in ["electoral", "popular vote", "270", "landslide", "margin", "sweep"]):
        return "electoral/margin"
    if any(k in ql for k in ["win the 2024 us pres", "win the presiden", "presidential election",
                             "be inaugurat", "win the election", "next president"]):
        return "presidential"
    if any(k in ql for k in ["cabinet", "secretary", "appoint", "confirm", "pardon"]):
        return "appointments"
    return "other"


def _subtopic_flow(net, codes, n_sub):
    members = [np.nonzero(codes == a)[0] for a in range(n_sub)]
    flow = np.zeros((n_sub, n_sub))
    for a in range(n_sub):
        for b in range(n_sub):
            if a != b and len(members[a]) and len(members[b]):
                flow[a, b] = net[np.ix_(members[a], members[b])].sum() / (len(members[a]) * len(members[b]))
    return flow


def main():
    log.info("=" * 60)
    log.info("Within-politics deep dive — sub-topic lead-lag")
    log.info("=" * 60)

    adj_df = pd.read_parquet(ADJACENCY_FILE)
    token_ids = list(adj_df.columns)
    adj = adj_df.to_numpy(dtype=float)
    meta = {r["token_id"]: r for r in load_json(ELIGIBLE_MARKETS_FILE)}

    pol_idx = np.array([i for i, t in enumerate(token_ids)
                        if meta.get(t, {}).get("category") == "Politics"])
    sub = np.array([_subtopic(meta[token_ids[i]]["question"]) for i in pol_idx])
    names = sorted(set(sub))
    code = {s: a for a, s in enumerate(names)}
    codes = np.array([code[s] for s in sub])
    n_sub = len(names)
    sizes = {s: int((sub == s).sum()) for s in names}
    log.info(f"Politics contracts: {len(pol_idx)} in {n_sub} sub-topics")
    for s in sorted(sizes, key=sizes.get, reverse=True):
        log.info(f"  {sizes[s]:3d}  {s}")

    net = (adj - adj.T)[np.ix_(pol_idx, pol_idx)]    # net flow on the politics subgraph
    flow = _subtopic_flow(net, codes, n_sub)
    lead = flow.sum(axis=1)

    # permutation null on sub-topic labels
    null_lead = np.empty((PERM, n_sub))
    null_edge = np.empty((PERM, n_sub, n_sub))
    for k in range(PERM):
        f = _subtopic_flow(net, _RNG.permutation(codes), n_sub)
        null_lead[k] = f.sum(axis=1)
        null_edge[k] = f
    lead_z = (lead - null_lead.mean(0)) / (null_lead.std(0) + 1e-12)

    log.info("\n── Sub-topic leadingness (leaders → followers) ──")
    rank = np.argsort(lead)[::-1]
    for a in rank:
        log.info(f"  {names[a]:16s}  L={lead[a]:+.4f}  z={lead_z[a]:+.2f}  (n={sizes[names[a]]})")

    # significant directed edges
    log.info("\n── Significant directed sub-topic edges (perm p<0.05) ──")
    edges = []
    for a in range(n_sub):
        for b in range(n_sub):
            if a == b:
                continue
            p = float((np.sum(null_edge[:, a, b] >= flow[a, b]) + 1) / (PERM + 1)) if flow[a, b] > 0 else 1.0
            if flow[a, b] > 0 and p < 0.05:
                edges.append({"lead": names[a], "lag": names[b], "net_flow": float(flow[a, b]), "p": p})
                log.info(f"  {names[a]} → {names[b]}: net_flow={flow[a,b]:+.4f}  p={p:.3f}")
    if not edges:
        log.info("  (none — no sub-topic significantly leads another)")

    ensure_dirs([PHASE5_DIR, PRESENTATION_DIR])
    save_json({"sizes": sizes, "subtopics": names,
               "leadingness": {names[a]: float(lead[a]) for a in range(n_sub)},
               "leadingness_z": {names[a]: float(lead_z[a]) for a in range(n_sub)},
               "flow_matrix": flow.tolist(), "significant_edges": edges},
              POLITICS_SUBFLOW_FILE)
    _figure(flow, lead, lead_z, names, sizes)
    log.info(f"\nSaved: {POLITICS_SUBFLOW_FILE}")
    return {"leadingness": lead, "edges": edges}


def _figure(flow, lead, lead_z, names, sizes):
    n_sub = len(names)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    g = nx.DiGraph()
    g.add_nodes_from(range(n_sub))
    for a in range(n_sub):
        for b in range(a + 1, n_sub):
            w = flow[a, b]
            if w > 0:
                g.add_edge(a, b, weight=w)
            elif w < 0:
                g.add_edge(b, a, weight=-w)
    pos = nx.circular_layout(nx.complete_graph(n_sub))
    node_sizes = [400 + 6000 * abs(lead[a]) for a in range(n_sub)]
    nodes = nx.draw_networkx_nodes(g, pos, node_size=node_sizes, node_color=lead,
                                   cmap="coolwarm", ax=ax1)
    if g.edges():
        ws = [g[u][v]["weight"] for u, v in g.edges()]
        wmax = max(ws)
        nx.draw_networkx_edges(g, pos, width=[1 + 5 * w / wmax for w in ws], edge_color="#555",
                               arrowsize=18, connectionstyle="arc3,rad=0.1", ax=ax1)
    nx.draw_networkx_labels(g, pos, {a: names[a] for a in range(n_sub)}, font_size=8, ax=ax1)
    fig.colorbar(nodes, ax=ax1, label="leadingness", fraction=0.046)
    ax1.set_title("(a) Within-politics meta-flow (edge = net lead→lag)")
    ax1.axis("off")

    order = np.argsort(lead)
    colors = ["#2b6cb0" if lead[a] > 0 else "#e53e3e" for a in order]
    ax2.barh(range(n_sub), [lead[a] for a in order], color=colors)
    ax2.set_yticks(range(n_sub))
    ax2.set_yticklabels([f"{names[a]} (n={sizes[names[a]]})" for a in order])
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("sub-topic leadingness")
    ax2.set_title("(b) Which political sub-topics lead")
    for i, a in enumerate(order):
        ax2.text(lead[a], i, f"  z={lead_z[a]:+.1f}", va="center", fontsize=8)
    fig.tight_layout()
    path = os.path.join(PRESENTATION_DIR, "12_politics_subflow.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
