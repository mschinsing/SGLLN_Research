"""
Step 2.1 — Contract semantic embedding
=======================================
Embeds each contract's QUESTION TEXT with Sentence-BERT (all-MiniLM-L6-v2 →
ê_i ∈ R^384) and builds the rectified-cosine semantic similarity matrix:

    W^sem_ij = [ ê_i · ê_j ]_+        (embeddings are L2-normalized → dot = cosine)

IMPORTANT (vs Phase 0 step 0.6): we embed the question text ONLY — no
"[Category: …]" / outcome suffix. The semantic signal must be independent of the
platform categories it is compared against in the three-way analysis (step 2.3),
otherwise the comparison is circular.

Node order is aligned to the Phase 1 adjacency/cluster order so every Phase 2
step shares one canonical ordering.

Input:  data/processed/eligible_markets.json
        data/processed/phase1/adjacency_matrix.parquet   (canonical node order)
Output: data/processed/phase2/contract_embeddings.npy    (n × 384, canonical order)
        data/processed/phase2/semantic_similarity.parquet (n × n, token-id index)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

from config import (
    ELIGIBLE_MARKETS_FILE, ADJACENCY_FILE,
    SEM_EMBEDDINGS_FILE, SEM_SIM_FILE, SEM_MODEL, PHASE2_DIR,
)
from utils import get_logger, ensure_dirs, load_json

log = get_logger("step2.1_semantic_embedding")


def _contract_text(rec):
    """Category-free text: question, else description, else slug words."""
    txt = (rec.get("question") or "").strip()
    if not txt:
        txt = (rec.get("description") or "").strip()
    if not txt:
        txt = (rec.get("market_slug") or "").replace("-", " ").strip()
    return txt


def main():
    log.info("=" * 60)
    log.info("STEP 2.1: Contract semantic embedding (category-free)")
    log.info("=" * 60)

    # Canonical node order = Phase 1 adjacency columns
    token_ids = list(pd.read_parquet(ADJACENCY_FILE).columns)
    recs = {r["token_id"]: r for r in load_json(ELIGIBLE_MARKETS_FILE)}
    texts = [_contract_text(recs[t]) for t in token_ids]
    log.info(f"Embedding {len(texts)} contract questions with {SEM_MODEL}")
    log.info(f"  sample: \"{texts[0][:70]}\"")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SEM_MODEL)
    emb = model.encode(texts, batch_size=64, show_progress_bar=False,
                       normalize_embeddings=True)
    emb = np.asarray(emb, dtype=float)
    log.info(f"Embedding matrix: {emb.shape}")

    # Rectified cosine similarity (unit-norm rows → dot product = cosine)
    sim = emb @ emb.T
    sim = np.maximum(sim, 0.0)
    np.fill_diagonal(sim, 0.0)

    ensure_dirs(PHASE2_DIR)
    np.save(SEM_EMBEDDINGS_FILE, emb)
    pd.DataFrame(sim, index=token_ids, columns=token_ids).to_parquet(SEM_SIM_FILE)
    log.info(f"Saved embeddings: {SEM_EMBEDDINGS_FILE}")
    log.info(f"Saved W^sem:      {SEM_SIM_FILE}")

    iu = np.triu_indices(len(token_ids), k=1)
    off = sim[iu]
    log.info("\n── SUMMARY ──")
    log.info(f"  W^sem mean/median: {off.mean():.4f} / {np.median(off):.4f}")
    log.info(f"  pairs > 0.5: {int((off > 0.5).sum()):,} | > 0.8: {int((off > 0.8).sum()):,}")

    return {"embeddings": emb, "similarity": sim, "token_ids": token_ids}


if __name__ == "__main__":
    main()
