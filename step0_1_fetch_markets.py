"""
Step 0.1 — Fetch market metadata from the Polymarket Gamma API
===============================================================
Paginates the Gamma /markets endpoint, keeps markets that were active during
the study window, and normalizes each into the snake_case schema that every
downstream step relies on.

Gamma returns camelCase fields and encodes `clobTokenIds` / `outcomes` as JSON
*strings*. This step is the single place that translation happens, so steps
0.2–0.6 can assume clean Python types.

Output: data/raw/markets_raw.json  — a list of dicts, each with keys:
    token-level downstream code reads:
        clob_token_ids (list[str]), outcomes (list[str]),
        condition_id, question, category, volume, market_slug,
        description, end_date, start_date
"""
import sys, os, json
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    GAMMA_URL, MARKETS_RAW_FILE, RAW_DIR,
    START_DATE, END_DATE,
    REQUEST_DELAY, MAX_RETRIES, RETRY_BACKOFF, REQUEST_TIMEOUT,
    GAMMA_PAGE_LIMIT, GAMMA_MAX_PAGES, GAMMA_OFFSET_MAX, GAMMA_VOLUME_MIN,
    CATEGORY_ALIASES,
)
from utils import get_logger, fetch_json, save_json, ensure_dirs

log = get_logger("step0.1_fetch_markets")


# --------------------------------------------------------------------------
# Field parsing helpers
# --------------------------------------------------------------------------
def parse_json_field(value):
    """Gamma encodes list fields as JSON strings. Return a real list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def parse_iso_date(value):
    """Parse an ISO-8601 timestamp to a date, or None."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.date()
    except (ValueError, TypeError):
        return None


# Gamma tags are numerous, unordered, and mix top-level themes with specific
# entities (e.g. the Trump market is tagged 'Trump', 'Politics', 'kamala harris',
# 'Elections', ...). For cluster-vs-category validation we want a single clean
# top-level label, so we match a market's tags against this priority-ordered
# canonical list and take the first hit. Order matters: earlier wins ties.
CANONICAL_CATEGORIES = [
    "Politics", "Elections", "Geopolitics", "Sports", "Crypto",
    "Business", "Economy", "Stocks", "Tech", "Science",
    "Climate", "Health", "Pop Culture", "Entertainment", "Celebrities",
]
# Tags that are navigational noise, not real categories.
GENERIC_TAGS = {"all", "potusbanner", "recurring", "weekly", "daily", "monthly"}


def _one_label(t):
    """Extract a single label string from a tag (dict or bare string)."""
    if isinstance(t, dict):
        return t.get("label") or t.get("slug")
    return t


def _tag_labels(m):
    """All tag label strings for a market, from its own `tags` (populated by
    include_tag=true) or, failing that, its parent event's tags."""
    labels = [str(l) for t in (m.get("tags") or []) if (l := _one_label(t))]
    if not labels:
        labels = [str(l) for ev in (m.get("events") or [])
                  for t in (ev.get("tags") or []) if (l := _one_label(t))]
    return labels


def _raw_category(m):
    """Best top-level category before alias normalization: an explicit category,
    else the first canonical theme present in tags, else the first non-generic tag."""
    cat = m.get("category")
    if cat:
        return str(cat)
    labels = _tag_labels(m)
    present = {l.lower() for l in labels}
    for canon in CANONICAL_CATEGORIES:
        if canon.lower() in present:
            return canon
    for l in labels:
        if l.lower() not in GENERIC_TAGS:
            return l
    return ""


def derive_category(m):
    """Single normalized top-level category for a market (stragglers folded via
    CATEGORY_ALIASES)."""
    cat = _raw_category(m)
    return CATEGORY_ALIASES.get(cat, cat)


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_market(m):
    """Map one raw Gamma market object into the downstream snake_case schema."""
    return {
        "condition_id":   m.get("conditionId") or m.get("condition_id") or "",
        "question":       m.get("question", "") or "",
        "description":    m.get("description", "") or "",
        "category":       derive_category(m),
        "market_slug":    m.get("slug") or m.get("market_slug") or "",
        "clob_token_ids": parse_json_field(m.get("clobTokenIds") or m.get("clob_token_ids")),
        "outcomes":       parse_json_field(m.get("outcomes")),
        "volume":         to_float(m.get("volumeNum", m.get("volume", 0))),
        "start_date":     (m.get("startDate") or m.get("start_date") or ""),
        "end_date":       (m.get("endDate") or m.get("end_date") or ""),
    }


def overlaps_window(market):
    """Keep markets whose [start, end] overlaps the study window at all."""
    s = parse_iso_date(market.get("start_date"))
    e = parse_iso_date(market.get("end_date"))
    # If a market has no end date, keep it and let the price-data filter decide.
    if e is None:
        return True
    if e < START_DATE:
        return False
    if s is not None and s > END_DATE:
        return False
    return True


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------
def fetch_all_markets():
    """Paginate Gamma /markets via limit/offset until a short page returns."""
    markets = []
    offset = 0
    for page in range(1, GAMMA_MAX_PAGES + 1):
        params = {
            "limit": GAMMA_PAGE_LIMIT,
            "offset": offset,
            "order": "volumeNum",
            "ascending": "false",
            "closed": "true",          # historical (resolved) markets
            "include_tag": "true",     # surface each market's tags (for category)
        }
        if GAMMA_VOLUME_MIN:
            params["volume_num_min"] = GAMMA_VOLUME_MIN

        try:
            batch = fetch_json(
                GAMMA_URL, params=params,
                max_retries=MAX_RETRIES, backoff=RETRY_BACKOFF,
                delay=REQUEST_DELAY, timeout=REQUEST_TIMEOUT, logger=log,
            )
        except RuntimeError as e:
            log.warning(f"  Stopped pagination at offset {offset}: {e}")
            break

        # Gamma returns a bare list; some deployments wrap it in {"data": [...]}
        items = batch if isinstance(batch, list) else batch.get("data", [])
        if not items:
            break

        markets.extend(items)
        log.info(f"  page {page}: +{len(items)} (total {len(markets)})")

        if len(items) < GAMMA_PAGE_LIMIT:
            break
        offset += GAMMA_PAGE_LIMIT
        if offset > GAMMA_OFFSET_MAX:
            log.info(
                f"  Reached Gamma offset ceiling ({GAMMA_OFFSET_MAX}); "
                f"stopping. Markets beyond this are the lowest-volume tail."
            )
            break

    return markets


def main():
    log.info("=" * 60)
    log.info("STEP 0.1: Fetch market metadata from Gamma API")
    log.info("=" * 60)
    log.info(f"Window: {START_DATE} -> {END_DATE}")

    ensure_dirs(RAW_DIR)

    raw = fetch_all_markets()
    log.info(f"Fetched {len(raw)} raw markets")

    normalized, kept = [], 0
    for m in raw:
        rec = normalize_market(m)
        if not rec["clob_token_ids"]:
            continue                       # unusable without token IDs
        if not overlaps_window(rec):
            continue
        normalized.append(rec)
        kept += 1

    save_json(normalized, MARKETS_RAW_FILE)
    log.info(f"Kept {kept} markets in window; saved to {MARKETS_RAW_FILE}")

    # quick category breakdown
    cats = {}
    for r in normalized:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    for cat, n in sorted(cats.items(), key=lambda kv: -kv[1])[:10]:
        log.info(f"  {cat or '(uncategorized)'}: {n}")

    return normalized


if __name__ == "__main__":
    main()
