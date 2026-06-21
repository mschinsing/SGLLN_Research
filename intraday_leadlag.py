"""
intraday_leadlag.py — lead-lag network at HOURLY resolution + daily comparison
==============================================================================
Re-runs the Phase-1 lead-lag pipeline on the hourly returns matrix (lags in
HOURS) and compares to the daily baseline. Answers: does measuring at the
timescale information actually flows change the structure, the topic-orthogonality,
or the within-politics cascade — and how many HOURS does information take to
propagate (a measurement impossible at daily resolution)?

Reuses the daily Phase-1 functions throughout. Same 354-contract universe.

Input:  data/processed/intraday/returns_hourly.parquet (+ daily baselines)
Output: data/processed/intraday/intraday_summary.json
        data/processed/presentation/13_intraday_vs_daily.png
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score
from joblib import Parallel, delayed

from config import (
    RETURNS_HOURLY_FILE, ADJACENCY_FILE, CLUSTERS_FILE, ELIGIBLE_MARKETS_FILE,
    INTRADAY_SUMMARY_FILE, INTRADAY_DIR, PRESENTATION_DIR,
    MAX_LAG_HOURS, MIN_OVERLAP_DAYS, PAIR_CHUNK_SIZE, N_JOBS, WF_K, WF_DEP_PCT,
)
from utils import get_logger, ensure_dirs, save_json, load_json
from step1_1_leadlag import _directional_integral, _abs_dcor
from step1_6_liquidity import _build_adjacency, _node_leadingness
from step1_4_spectral import _normalized_hermitian, _embeddings, _cluster
from step1_5_metaflow import _meta_flow
from politics_deepdive import _subtopic

log = get_logger("intraday_leadlag")
MIN_HOURS = 24 * MIN_OVERLAP_DAYS     # require >= same calendar overlap as daily, in hours


def _idcor_chunk(chunk, R, L):
    out = []
    for i, j in chunk:
        ri, rj = R[:, i], R[:, j]
        if ((~np.isnan(ri)) & (~np.isnan(rj))).sum() < MIN_HOURS:
            continue
        out.append((i, j, _directional_integral(ri, rj, L, _abs_dcor),
                    _directional_integral(rj, ri, L, _abs_dcor)))
    return out


def _idcor_matrix(R, pairs, L):
    n = R.shape[1]
    chunks = [pairs[k:k + PAIR_CHUNK_SIZE] for k in range(0, len(pairs), PAIR_CHUNK_SIZE)]
    res = Parallel(n_jobs=N_JOBS)(delayed(_idcor_chunk)(c, R, L) for c in chunks)
    mat = np.zeros((n, n))
    for rows in res:
        for i, j, a, b in rows:
            mat[i, j] = a
            mat[j, i] = b
    return mat


def _matched_tau(idc, n_edges):
    n = idc.shape[0]
    m = np.maximum(idc, idc.T)[np.triu_indices(n, 1)]
    return float(np.quantile(m, 1 - n_edges / len(m)))


def _hermitian(adj, k):
    a_sym, _, _ = _normalized_hermitian(adj)
    ev, V = eigh(a_sym)
    return _cluster(_embeddings(ev, V, k, True), k)


def _dominant_lag(ri, rj, L):
    best_l, best = 0, -1.0
    n = len(ri)
    for l in range(1, L + 1):
        x, y = ri[:n - l], rj[l:]
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() < MIN_HOURS:
            continue
        d = _abs_dcor(x[m], y[m])
        if d > best:
            best, best_l = d, l
    return best_l


def main():
    log.info("=" * 60)
    log.info(f"INTRADAY lead-lag (hourly, lags 1..{MAX_LAG_HOURS}h) vs daily baseline")
    log.info("=" * 60)

    rdf = pd.read_parquet(RETURNS_HOURLY_FILE)
    token_ids = list(rdf.columns)
    R = rdf.to_numpy(dtype=float)
    n = len(token_ids)
    log.info(f"Hourly returns: {R.shape}")

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    t0 = time.time()
    log.info(f"Computing hourly I_dcor over {len(pairs):,} pairs × {MAX_LAG_HOURS} lags...")
    idc = _idcor_matrix(R, pairs, MAX_LAG_HOURS)
    log.info(f"  done in {time.time()-t0:.0f}s")

    # daily baseline
    adj_d = pd.read_parquet(ADJACENCY_FILE).to_numpy(dtype=float)
    n_edges_d = int((adj_d > 0).sum())
    lead_d = _node_leadingness(adj_d)
    labels_d = np.array([load_json(CLUSTERS_FILE)["labels"][t] for t in token_ids])

    # hourly adjacency, matched to daily edge count for fair comparison
    adj_h = _build_adjacency(idc, _matched_tau(idc, n_edges_d))
    lead_h = _node_leadingness(adj_h)
    labels_h = _hermitian(adj_h, WF_K)

    cats = np.array([{r["token_id"]: r.get("category", "")
                      for r in load_json(ELIGIBLE_MARKETS_FILE)}.get(t, "") for t in token_ids])
    _, cat_codes = np.unique(cats, return_inverse=True)

    # ── Comparisons ──────────────────────────────────────────────────────
    eo, eh = adj_d > 0, adj_h > 0
    edge_overlap = int((eo & eh).sum()) / max(int(eo.sum()), 1)
    lead_rho = float(spearmanr(lead_d, lead_h).correlation)
    ari_hd = float(adjusted_rand_score(labels_d, labels_h))
    ari_hcat = float(adjusted_rand_score(cat_codes, labels_h))
    ari_dcat = float(adjusted_rand_score(cat_codes, labels_d))

    log.info("\n── HOURLY vs DAILY ──")
    log.info(f"  edge overlap (daily edges also hourly): {100*edge_overlap:.1f}%")
    log.info(f"  node-leadingness Spearman (hourly vs daily): {lead_rho:+.3f}")
    log.info(f"  clusters: ARI(hourly,daily)={ari_hd:.3f}  ARI(hourly,category)={ari_hcat:.3f} "
             f"(daily was {ari_dcat:.3f})  → orthogonality {'HOLDS' if ari_hcat<0.2 else 'CHANGES'}")

    # ── Propagation speed: dominant lag (hours) over surviving directed edges ──
    net = adj_h - adj_h.T
    lead_edges = [(i, j) for i in range(n) for j in range(n) if net[i, j] > 0]
    lags = [_dominant_lag(R[:, i], R[:, j], MAX_LAG_HOURS) for i, j in lead_edges]
    lags = [l for l in lags if l > 0]
    log.info(f"\n── PROPAGATION SPEED ──")
    log.info(f"  dominant lag over {len(lags)} directed edges: "
             f"median={np.median(lags):.0f}h  mean={np.mean(lags):.1f}h  "
             f"(≤3h: {100*np.mean(np.array(lags)<=3):.0f}%)")

    # ── Within-politics cascade at hourly ────────────────────────────────
    pol = np.array([i for i, t in enumerate(token_ids)
                    if {r["token_id"]: r.get("category", "")
                        for r in load_json(ELIGIBLE_MARKETS_FILE)}.get(t) == "Politics"])
    sub = np.array([_subtopic({r["token_id"]: r for r in load_json(ELIGIBLE_MARKETS_FILE)}
                              [token_ids[i]]["question"]) for i in pol])
    snames = sorted(set(sub))
    scode = np.array([snames.index(s) for s in sub])
    subnet = (adj_h - adj_h.T)[np.ix_(pol, pol)]
    members = [np.nonzero(scode == a)[0] for a in range(len(snames))]
    sub_lead = {}
    for a, nm in enumerate(snames):
        outflow = sum(subnet[np.ix_(members[a], members[b])].sum() / (len(members[a])*len(members[b]))
                      for b in range(len(snames)) if b != a and len(members[a]) and len(members[b]))
        sub_lead[nm] = float(outflow)
    log.info("\n── Within-politics leadingness (hourly) ──")
    for nm in sorted(sub_lead, key=sub_lead.get, reverse=True):
        log.info(f"  {nm:16s} L={sub_lead[nm]:+.4f}")

    summary = {"max_lag_hours": MAX_LAG_HOURS, "n_hours": R.shape[0],
               "edge_overlap_with_daily": edge_overlap, "leadingness_spearman": lead_rho,
               "ari_hourly_daily": ari_hd, "ari_hourly_category": ari_hcat,
               "ari_daily_category": ari_dcat,
               "dominant_lag_median_h": float(np.median(lags)), "dominant_lag_mean_h": float(np.mean(lags)),
               "subtopic_leadingness_hourly": sub_lead}
    ensure_dirs([INTRADAY_DIR, PRESENTATION_DIR])
    save_json(summary, INTRADAY_SUMMARY_FILE)
    _figure(lead_d, lead_h, lags, ari_dcat, ari_hcat, lead_rho, edge_overlap, sub_lead)
    log.info(f"\nSaved: {INTRADAY_SUMMARY_FILE}")
    return summary


def _figure(lead_d, lead_h, lags, ari_dcat, ari_hcat, rho, overlap, sub_lead):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))
    ax1.scatter(lead_d, lead_h, s=14, alpha=0.5, color="#2b6cb0")
    lim = max(np.abs(lead_d).max(), np.abs(lead_h).max()) * 1.05
    ax1.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax1.set_xlabel("daily leadingness"); ax1.set_ylabel("hourly leadingness")
    ax1.set_title(f"(a) Leadingness daily vs hourly (ρ={rho:+.2f})")

    ax2.hist(lags, bins=range(1, max(lags) + 2), color="#dd6b20", align="left")
    ax2.axvline(np.median(lags), color="red", ls="--", label=f"median {np.median(lags):.0f}h")
    ax2.set_xlabel("dominant lag (hours)"); ax2.set_ylabel("directed edges")
    ax2.set_title("(b) Information propagation speed"); ax2.legend(fontsize=9)

    items = sorted(sub_lead.items(), key=lambda kv: kv[1])
    nm, vals = zip(*items)
    ax3.barh(range(len(nm)), vals, color=["#2b6cb0" if v > 0 else "#e53e3e" for v in vals])
    ax3.set_yticks(range(len(nm))); ax3.set_yticklabels(nm, fontsize=8)
    ax3.axvline(0, color="black", lw=0.8)
    ax3.set_title(f"(c) Within-politics cascade (hourly)\northogonality: daily ARI {ari_dcat:.2f} → hourly {ari_hcat:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(PRESENTATION_DIR, "13_intraday_vs_daily.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
