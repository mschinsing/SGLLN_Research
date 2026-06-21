"""
Step 0.6 — Supplementary data: event calendar & semantic embeddings
=====================================================================
1. Curated event calendar (FOMC, CPI, debates, jobs reports)
2. Contract description extraction and sentence-transformer embedding

Input:  data/processed/eligible_markets.json
Output: data/processed/event_calendar.csv
        data/processed/contract_descriptions.json
        data/processed/embeddings/contract_embeddings.npy
        data/processed/embeddings/token_ids.json
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from config import (
    EVENTS_FILE, DESCRIPTIONS_FILE, EMBEDDINGS_DIR,
    ELIGIBLE_MARKETS_FILE, PROCESSED_DIR,
)
from utils import get_logger, load_json, save_json, ensure_dirs

log = get_logger("step0.6_supplementary")

# ── Curated event calendar: June 1 – Nov 5, 2024 ─────────────────────
# Sources: Federal Reserve calendar, BLS release schedule, debate schedule
EVENT_CALENDAR = [
    # FOMC meetings (announcement dates)
    {"date": "2024-06-12", "type": "FOMC", "description": "FOMC meeting — rate decision (June)"},
    {"date": "2024-07-31", "type": "FOMC", "description": "FOMC meeting — rate decision (July)"},
    {"date": "2024-09-18", "type": "FOMC", "description": "FOMC meeting — rate decision (September)"},
    {"date": "2024-11-07", "type": "FOMC", "description": "FOMC meeting — rate decision (November)"},

    # CPI releases
    {"date": "2024-06-12", "type": "CPI", "description": "CPI release (May data)"},
    {"date": "2024-07-11", "type": "CPI", "description": "CPI release (June data)"},
    {"date": "2024-08-14", "type": "CPI", "description": "CPI release (July data)"},
    {"date": "2024-09-11", "type": "CPI", "description": "CPI release (August data)"},
    {"date": "2024-10-10", "type": "CPI", "description": "CPI release (September data)"},

    # Jobs reports (non-farm payrolls, first Friday of month)
    {"date": "2024-06-07", "type": "JOBS", "description": "Non-farm payrolls (May data)"},
    {"date": "2024-07-05", "type": "JOBS", "description": "Non-farm payrolls (June data)"},
    {"date": "2024-08-02", "type": "JOBS", "description": "Non-farm payrolls (July data)"},
    {"date": "2024-09-06", "type": "JOBS", "description": "Non-farm payrolls (August data)"},
    {"date": "2024-10-04", "type": "JOBS", "description": "Non-farm payrolls (September data)"},
    {"date": "2024-11-01", "type": "JOBS", "description": "Non-farm payrolls (October data)"},

    # Presidential debates
    {"date": "2024-06-27", "type": "DEBATE", "description": "Presidential debate — Biden vs Trump (CNN)"},
    {"date": "2024-09-10", "type": "DEBATE", "description": "Presidential debate — Harris vs Trump (ABC)"},
    {"date": "2024-10-01", "type": "DEBATE", "description": "VP debate — Walz vs Vance (CBS)"},

    # Other major political/economic events
    {"date": "2024-07-13", "type": "POLITICAL", "description": "Trump assassination attempt (Butler, PA)"},
    {"date": "2024-07-21", "type": "POLITICAL", "description": "Biden withdraws from presidential race"},
    {"date": "2024-08-19", "type": "POLITICAL", "description": "DNC begins (Chicago)"},
    {"date": "2024-07-15", "type": "POLITICAL", "description": "RNC begins (Milwaukee)"},
    {"date": "2024-11-05", "type": "ELECTION", "description": "2024 US General Election Day"},
]


def save_event_calendar():
    """Save the curated event calendar as CSV."""
    ensure_dirs(os.path.dirname(EVENTS_FILE))

    with open(EVENTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "type", "description"])
        writer.writeheader()
        for event in sorted(EVENT_CALENDAR, key=lambda e: e["date"]):
            writer.writerow(event)

    log.info(f"Saved event calendar: {EVENTS_FILE} ({len(EVENT_CALENDAR)} events)")


def extract_descriptions(eligible: list) -> dict:
    """
    Extract and clean contract descriptions for semantic embedding.
    Uses 'question' as primary text, falling back to 'description'.
    De-duplicates by condition_id (multiple tokens per market share text).
    """
    descriptions = {}
    seen_conditions = set()

    for rec in eligible:
        tid = rec["token_id"]
        cid = rec.get("condition_id", tid)

        # Primary text: question (short, specific)
        text = rec.get("question", "").strip()
        if not text:
            text = rec.get("description", "").strip()
        if not text:
            text = rec.get("market_slug", "").replace("-", " ")

        # Append category and outcome for richer semantics
        category = rec.get("category", "")
        outcome_idx = rec.get("outcome_index", 0)
        suffix_parts = []
        if category:
            suffix_parts.append(f"[Category: {category}]")
        if outcome_idx == 0:
            suffix_parts.append("[YES outcome]")
        else:
            suffix_parts.append("[NO outcome]")

        full_text = f"{text} {' '.join(suffix_parts)}".strip()

        descriptions[tid] = {
            "token_id": tid,
            "condition_id": cid,
            "text": full_text,
            "question": rec.get("question", ""),
            "category": category,
        }

    return descriptions


def compute_embeddings(descriptions: dict) -> tuple:
    """
    Embed contract descriptions using sentence-transformers.
    Uses all-MiniLM-L6-v2 (22M params, CPU-friendly, 384-dim output).

    Returns:
        embeddings: np.ndarray of shape (n_tokens, 384)
        token_ids: list of token ID strings (in order)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.warning(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers\n"
            "Skipping embeddings — will save placeholder."
        )
        return None, None

    model_name = "all-MiniLM-L6-v2"
    log.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    token_ids = list(descriptions.keys())
    texts = [descriptions[tid]["text"] for tid in token_ids]

    log.info(f"Embedding {len(texts)} contract descriptions...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # Unit-norm for cosine similarity
    )

    log.info(f"Embedding shape: {embeddings.shape}")  # (n, 384)
    return embeddings, token_ids


def compute_semantic_similarity(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise semantic similarity matrix.
    W^sem_ij = [e_i^T e_j]_+  (rectified cosine similarity).
    Since embeddings are L2-normalized, dot product = cosine similarity.
    """
    sim = embeddings @ embeddings.T  # (n, n)
    sim = np.maximum(sim, 0)  # rectify: keep only positive similarities
    np.fill_diagonal(sim, 0)  # zero self-similarity
    return sim


def main():
    log.info("=" * 60)
    log.info("STEP 0.6: Supplementary data — events & embeddings")
    log.info("=" * 60)

    # ── Event calendar ───────────────────────────────────────────────
    save_event_calendar()

    # ── Contract descriptions ────────────────────────────────────────
    eligible = load_json(ELIGIBLE_MARKETS_FILE)
    log.info(f"Loaded {len(eligible)} eligible markets")

    descriptions = extract_descriptions(eligible)
    save_json(descriptions, DESCRIPTIONS_FILE)
    log.info(f"Saved {len(descriptions)} descriptions: {DESCRIPTIONS_FILE}")

    # Show sample
    for i, (tid, desc) in enumerate(descriptions.items()):
        if i >= 3:
            break
        log.info(f"  Sample: [{desc['category']}] {desc['text'][:80]}...")

    # ── Semantic embeddings ──────────────────────────────────────────
    ensure_dirs(EMBEDDINGS_DIR)

    embeddings, token_ids = compute_embeddings(descriptions)

    if embeddings is not None:
        # Save embeddings
        emb_path = os.path.join(EMBEDDINGS_DIR, "contract_embeddings.npy")
        np.save(emb_path, embeddings)
        log.info(f"Saved embeddings: {emb_path}")

        ids_path = os.path.join(EMBEDDINGS_DIR, "token_ids.json")
        save_json(token_ids, ids_path)
        log.info(f"Saved token ID order: {ids_path}")

        # Compute and save semantic similarity matrix
        log.info("Computing semantic similarity matrix...")
        sim_matrix = compute_semantic_similarity(embeddings)
        sim_path = os.path.join(EMBEDDINGS_DIR, "semantic_similarity.npy")
        np.save(sim_path, sim_matrix)
        log.info(f"Saved semantic similarity: {sim_path}")

        # Summary stats
        upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        log.info(f"  Mean sim: {np.mean(upper_tri):.4f}")
        log.info(f"  Median sim: {np.median(upper_tri):.4f}")
        log.info(f"  Max sim: {np.max(upper_tri):.4f}")
        log.info(f"  Pairs with sim > 0.5: {np.sum(upper_tri > 0.5)}")
        log.info(f"  Pairs with sim > 0.8: {np.sum(upper_tri > 0.8)}")
    else:
        log.warning("Embeddings skipped — placeholder files only")
        save_json(list(descriptions.keys()), os.path.join(EMBEDDINGS_DIR, "token_ids.json"))

    log.info("STEP 0.6 COMPLETE")


if __name__ == "__main__":
    main()
