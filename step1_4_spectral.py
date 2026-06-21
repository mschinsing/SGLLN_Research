"""
Step 1.4 — Hermitian random-walk spectral clustering
====================================================
Clusters the directed lead–lag network via the Hermitian (magnetic-Laplacian)
construction, which encodes edge *direction* as complex phase:

  1. flow   = A − Aᵀ                          (real skew-symmetric net flow)
  2. Ã      = i · flow                         (purely imaginary → Hermitian, real eigenvalues)
  3. Ã_sym  = D^{-1/2} Ã D^{-1/2}              (D = |Ã| row sums; D^{-1/2}=0 for isolated nodes)
  4. eigh(Ã_sym) → real eigenvalues (symmetric about 0, in ± pairs) + complex eigenvectors
  5. choose k by the eigengap on the positive spectrum; validate eigenvalue
     significance with a permutation test (PERM_B reps that shuffle the flow)
  6. node embedding v_i ∈ R^{2k} = concat(Re, Im) of the top-k positive-eigenvalue
     eigenvectors (one per ± pair)
  7. KMeans(emb, k) → clusters; sweep k ∈ K_VALUES for sensitivity

Diagnostic preview: ARI / AMI of the clusters vs platform categories (the
chance-corrected metrics chosen for the cluster-vs-category validation).

Input:  data/processed/phase1/adjacency_matrix.parquet
        data/processed/eligible_markets.json   (categories, for the ARI/AMI preview)
Output: data/processed/phase1/clusters.json
        data/processed/phase1/eigenvalues.npy
        data/processed/phase1/node_embeddings.npy
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Pin OpenMP to 1 thread BEFORE importing sklearn: KMeans otherwise calls
# threadpoolctl introspection that crashes on this conda/MKL build, and
# threading is irrelevant for a 354-node problem.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    ADJACENCY_FILE, ELIGIBLE_MARKETS_FILE,
    CLUSTERS_FILE, EIGEN_FILE, NODE_EMBEDDINGS_FILE, PHASE1_DIR, FIGURES_DIR,
    K_VALUES, EIGENGAP_KMIN, PERM_B, RANDOM_SEED, EMBED_ROW_NORMALIZE,
)
from utils import get_logger, ensure_dirs, save_json, load_json

log = get_logger("step1.4_spectral")

POS_EPS = 1e-9          # treat |eigenvalue| below this as zero
N_EIG_TEST = 10         # how many top positive eigenvalues to significance-test


def _normalized_hermitian(adj):
    """Return (Ã_sym, flow, degree). Isolated nodes get D^{-1/2}=0 (no div-by-zero)."""
    flow = adj - adj.T                      # real skew-symmetric
    a_tilde = 1j * flow                     # Hermitian
    degree = np.abs(a_tilde).sum(axis=1)    # = |flow| row sums
    d_inv_sqrt = np.zeros_like(degree)      # isolated nodes stay 0 (no div-by-zero)
    nz = degree > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(degree[nz])
    a_sym = d_inv_sqrt[:, None] * a_tilde * d_inv_sqrt[None, :]
    return a_sym, flow, degree


def _top_positive(eigvals, count):
    """Indices of the `count` largest positive eigenvalues (descending)."""
    order = np.argsort(eigvals)[::-1]
    pos = [i for i in order if eigvals[i] > POS_EPS]
    return pos[:count]


def _embeddings(eigvals, eigvecs, k, row_normalize):
    """R^{2k} node embedding: concat(Re, Im) of the top-k positive-eigenvalue eigenvectors.
    If row_normalize, L2-normalize each node's row (NJW) — guards zero rows (isolated nodes)."""
    idx = _top_positive(eigvals, k)
    vk = eigvecs[:, idx]                     # (n, k) complex
    emb = np.concatenate([vk.real, vk.imag], axis=1)
    if row_normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
    return emb


def _eigengap_k(eigvals, kmin=2, kmax=10):
    """k at the largest gap in the descending positive spectrum, within [kmin, kmax]."""
    pos = np.sort(eigvals[eigvals > POS_EPS])[::-1]
    if len(pos) < kmin + 1:
        return kmin
    gaps = pos[:-1] - pos[1:]
    hi = min(kmax, len(gaps))
    return int(np.argmax(gaps[kmin - 1:hi]) + kmin)


def _cluster(emb, k):
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    return km.fit_predict(emb)


def _plot_spectrum(eigvals, k_star, k_values, out_path, n_show=15):
    """Scree + eigengap plot to make the k choice visually defensible.
    Top: top positive eigenvalues. Bottom: gap λ_k − λ_{k+1} at each k; the
    eigengap heuristic picks the tallest bar inside the allowed [min,max] window."""
    pos = np.sort(eigvals[eigvals > POS_EPS])[::-1][:n_show]
    gaps = pos[:-1] - pos[1:]
    ks_eig = np.arange(1, len(pos) + 1)
    ks_gap = np.arange(1, len(gaps) + 1)            # gap at k = λ_k − λ_{k+1}
    kmin, kmax = min(k_values), max(k_values)
    dominant_k = int(ks_gap[np.argmax(gaps)])        # globally largest gap

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(ks_eig, pos, "o-", color="#2b6cb0")
    ax1.axvline(k_star, color="red", ls="--", label=f"selected k={k_star} (eigengap in [{kmin},{kmax}])")
    ax1.set_ylabel("positive eigenvalue λ")
    ax1.set_title("Hermitian spectrum (top positive eigenvalues)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    colors = ["#e53e3e" if k == k_star else ("#dd6b20" if k == dominant_k else "#a0aec0")
              for k in ks_gap]
    ax2.bar(ks_gap, gaps, color=colors)
    ax2.axvspan(kmin - 0.5, kmax + 0.5, color="blue", alpha=0.06,
                label=f"search window [{kmin},{kmax}]")
    ax2.set_xlabel("k (number of clusters)")
    ax2.set_ylabel("eigengap  λ_k − λ_{k+1}")
    ax2.set_title(f"Eigengaps — dominant gap at k={dominant_k}; selected k={k_star}")
    ax2.set_xticks(ks_gap)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return dominant_k


def _permutation_eig_test(flow, observed, n_reps, seed):
    """Null: shuffle the skew-symmetric flow's upper triangle (destroys topology,
    preserves the flow-magnitude distribution). Returns one-sided p-values for each
    of the top observed positive eigenvalues."""
    rng = np.random.default_rng(seed)
    n = flow.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = flow[iu]
    m = len(observed)
    null_top = np.zeros((n_reps, m))
    for b in range(n_reps):
        wp = np.zeros_like(flow)
        wp[iu] = rng.permutation(vals)
        wp = wp - wp.T
        a_tilde = 1j * wp
        deg = np.abs(a_tilde).sum(axis=1)
        d = np.zeros_like(deg)
        nz = deg > 0
        d[nz] = 1.0 / np.sqrt(deg[nz])
        a_sym = d[:, None] * a_tilde * d[None, :]
        ev = eigh(a_sym, eigvals_only=True)
        pos = np.sort(ev[ev > POS_EPS])[::-1]
        null_top[b, :min(m, len(pos))] = pos[:min(m, len(pos))]
    return [float((np.sum(null_top[:, r] >= observed[r]) + 1) / (n_reps + 1))
            for r in range(m)]


def main():
    log.info("=" * 60)
    log.info("STEP 1.4: Hermitian random-walk spectral clustering")
    log.info("=" * 60)

    adj_df = pd.read_parquet(ADJACENCY_FILE)
    token_ids = list(adj_df.columns)
    adj = adj_df.to_numpy(dtype=float)
    n = adj.shape[0]

    a_sym, flow, degree = _normalized_hermitian(adj)
    n_isolated = int((degree == 0).sum())
    log.info(f"Network: {n} nodes, {int((adj>0).sum()):,} edges, {n_isolated} isolated")

    eigvals, eigvecs = eigh(a_sym)          # real eigenvalues asc, complex eigenvectors
    np.save(EIGEN_FILE, eigvals)

    # Eigengap-selected k + permutation significance of the top eigenvalues
    # Hierarchical k: the eigengap with its real floor (k>=2) gives the DOMINANT split;
    # the eigengap restricted to the sensitivity range gives the FINER structure used
    # for the meta-flow / liquidity analysis downstream.
    k_dominant = _eigengap_k(eigvals, kmin=EIGENGAP_KMIN, kmax=max(K_VALUES))
    k_fine = _eigengap_k(eigvals, kmin=min(K_VALUES), kmax=max(K_VALUES))
    k_star = k_fine                          # downstream (1.5/1.6) uses the finer clustering
    top_idx = _top_positive(eigvals, N_EIG_TEST)
    observed_top = eigvals[top_idx]
    log.info(f"Top positive eigenvalues: {np.round(observed_top, 3).tolist()}")

    ensure_dirs([PHASE1_DIR, FIGURES_DIR])
    spectrum_path = os.path.join(FIGURES_DIR, "eigenvalue_spectrum.png")
    eval_ks = sorted(set(K_VALUES) | {k_dominant, k_fine})
    _plot_spectrum(eigvals, k_fine, eval_ks, spectrum_path)
    log.info(f"Eigengap (floor k={EIGENGAP_KMIN}) → DOMINANT k = {k_dominant}; "
             f"finer structure within [{min(K_VALUES)},{max(K_VALUES)}] → k = {k_fine}")
    log.info(f"Saved spectrum plot: {spectrum_path}")
    log.info(f"Running permutation test ({PERM_B} reps) on top {len(observed_top)} eigenvalues...")
    pvals = _permutation_eig_test(flow, observed_top, PERM_B, RANDOM_SEED)
    n_sig = int(np.sum(np.array(pvals) < 0.05))
    log.info(f"Significant eigenvalues (p<0.05): {n_sig}/{len(pvals)}  p-values={np.round(pvals,3).tolist()}")

    # Category labels for the ARI/AMI preview
    meta = {r["token_id"]: r.get("category", "") for r in load_json(ELIGIBLE_MARKETS_FILE)}
    cats = np.array([meta.get(t, "") for t in token_ids])
    _, cat_codes = np.unique(cats, return_inverse=True)

    # Sensitivity sweep over K_VALUES, comparing raw vs row-normalized embeddings.
    # The config flag EMBED_ROW_NORMALIZE decides which variant is saved.
    def _eval(k, row_norm):
        emb = _embeddings(eigvals, eigvecs, k, row_norm)
        labels = _cluster(emb, k)
        return (emb, labels,
                adjusted_rand_score(cat_codes, labels),
                adjusted_mutual_info_score(cat_codes, labels),
                np.bincount(labels, minlength=k).tolist())

    log.info("Comparing embeddings (raw concat vs NJW row-normalized):")
    labels_by_k = {}
    primary_labels = None
    for k in eval_ks:
        _, lab_raw, ari_r, ami_r, sz_r = _eval(k, False)
        emb_n, lab_norm, ari_n, ami_n, sz_n = _eval(k, True)
        flag = "  ← selected k" if k == k_star else ""
        log.info(f"  k={k:2d} raw     : ARI={ari_r:+.3f} AMI={ami_r:+.3f} sizes={sz_r}")
        log.info(f"  k={k:2d} rownorm : ARI={ari_n:+.3f} AMI={ami_n:+.3f} sizes={sz_n}{flag}")
        # Save the config-selected variant's labels
        chosen = lab_norm if EMBED_ROW_NORMALIZE else lab_raw
        labels_by_k[str(k)] = {t: int(c) for t, c in zip(token_ids, chosen)}
        if k == k_star:
            primary_labels = chosen
            np.save(NODE_EMBEDDINGS_FILE, emb_n if EMBED_ROW_NORMALIZE else _embeddings(eigvals, eigvecs, k, False))

    ensure_dirs(PHASE1_DIR)
    save_json({
        "k_selected": k_star,
        "k_dominant": k_dominant,
        "k_fine": k_fine,
        "k_values": eval_ks,
        "row_normalized": EMBED_ROW_NORMALIZE,
        "eigenvalue_pvalues": pvals,
        "n_significant_eigs": n_sig,
        "n_isolated_nodes": n_isolated,
        "labels": labels_by_k[str(k_star)],
        "labels_by_k": labels_by_k,
    }, CLUSTERS_FILE)
    log.info(f"\nSaved clusters: {CLUSTERS_FILE}")
    log.info(f"Saved eigenvalues: {EIGEN_FILE}")
    log.info(f"Saved node embeddings (k={k_star}): {NODE_EMBEDDINGS_FILE}")

    # Hierarchical summary: dominant split + finer structure
    dom_sizes = np.bincount([labels_by_k[str(k_dominant)][t] for t in token_ids])
    fine_sizes = np.bincount(primary_labels)
    log.info("\n── HIERARCHICAL STRUCTURE ──")
    log.info(f"  DOMINANT  k={k_dominant} (largest eigengap): sizes={dom_sizes.tolist()}")
    log.info(f"  FINER     k={k_fine} (downstream meta-flow): sizes={fine_sizes.tolist()}")

    return {"k_selected": k_star, "k_dominant": k_dominant, "labels": primary_labels,
            "token_ids": token_ids, "eigvals": eigvals}


if __name__ == "__main__":
    main()
