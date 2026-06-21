"""
utils.py
========
Shared helpers for the Phase 0 pipeline. Signatures are matched to how the
step files call them:

  ensure_dirs(path_or_paths)   # called with a single string path in the steps
  get_logger(name)
  fetch_json(url, params=, max_retries=, backoff=, delay=, timeout=, logger=)
       -> parsed JSON; raises RuntimeError once retries are exhausted
          (step0_3 specifically catches RuntimeError)
  save_json(obj, path)
  load_json(path)
"""

import os
import sys
import json
import time
import logging

import requests

# Make `import config` work from any of the sibling step files.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# One reused session for connection pooling across all requests.
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "phase0-research-pipeline/0.1"})


# ==========================================================================
# Directories
# ==========================================================================
def ensure_dirs(paths):
    """Create one directory or a list of directories (idempotent).

    Accepts either a single path string or an iterable of path strings,
    because the steps call it both ways.
    """
    if isinstance(paths, (str, bytes, os.PathLike)):
        paths = [paths]
    created = []
    for p in paths:
        if p:
            os.makedirs(p, exist_ok=True)
            created.append(p)
    return created


# ==========================================================================
# Logging
# ==========================================================================
def get_logger(name="phase0", level=logging.INFO):
    """Return a configured stdout logger (no duplicate handlers on re-import)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ==========================================================================
# HTTP
# ==========================================================================
def fetch_json(url, params=None, max_retries=4, backoff=1.5,
               delay=0.5, timeout=30, logger=None):
    """GET `url` with `params`, returning parsed JSON.

    Retries on network errors, timeouts, 429s, and 5xx responses with
    exponential backoff (wait = backoff ** attempt). Client errors (4xx,
    except 429) are permanent and fail fast. After `max_retries` failed
    attempts, raises RuntimeError — step0_3 relies on this type.
    """
    def _backoff(attempt, reason):
        wait = backoff ** attempt
        if logger:
            logger.warning(
                f"{reason} (attempt {attempt}/{max_retries}); retrying in {wait:.1f}s"
            )
        time.sleep(wait)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = _SESSION.get(url, params=params, timeout=timeout)

            # Rate-limit / transient server errors: retry with backoff.
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = RuntimeError(f"HTTP {resp.status_code}")
                _backoff(attempt, f"HTTP {resp.status_code} on {url}")
                continue

            # Client errors (4xx, e.g. 422 offset-too-large) are permanent —
            # retrying is pointless, so fail fast and let the caller handle it.
            if 400 <= resp.status_code < 500:
                raise RuntimeError(
                    f"HTTP {resp.status_code} (client error) for {url}: {resp.text[:200]}"
                )

            data = resp.json()
            if delay:
                time.sleep(delay)            # be polite to public endpoints
            return data

        except (requests.RequestException, ValueError) as e:
            last_err = e
            _backoff(attempt, f"Request error on {url}: {e}")

    raise RuntimeError(
        f"fetch_json failed after {max_retries} attempts for {url}: {last_err}"
    )


# ==========================================================================
# JSON IO
# ==========================================================================
def save_json(obj, path):
    """Write `obj` as pretty JSON, creating parent dirs as needed."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def load_json(path):
    """Read and parse a JSON file."""
    with open(path) as f:
        return json.load(f)


# ==========================================================================
# Time-series resampling (Phase 1)
# ==========================================================================
def stationary_block_bootstrap_indices(n, mean_block_len, rng):
    """Politis & Romano (1994) stationary block bootstrap index sequence.

    Returns an int array of length `n` indexing into a series of length `n`.
    Block lengths are Geometric(p = 1/mean_block_len) so the expected block
    length is `mean_block_len`; each block starts at a uniform-random position
    and wraps around circularly. Resampling a series with these indices
    preserves within-series autocorrelation while randomizing absolute timing.

    `rng` is a numpy Generator (np.random.default_rng) for reproducibility.
    """
    import numpy as np
    mean_block_len = max(1.0, float(mean_block_len))
    p = 1.0 / mean_block_len
    idx = np.empty(n, dtype=np.int64)
    filled = 0
    while filled < n:
        start = rng.integers(0, n)
        block_len = rng.geometric(p)               # >= 1
        take = min(block_len, n - filled)
        positions = (start + np.arange(take)) % n   # circular wrap
        idx[filled:filled + take] = positions
        filled += take
    return idx


def mean_acf_block_length(returns_df):
    """Tune an expected block length from the data's lag-1 autocorrelation.

    Takes the median lag-1 autocorrelation across all columns (ignoring NaNs)
    and maps it to an expected block length b = max(1, round(1/(1-rho))).
    Stronger persistence -> longer blocks. Returns a float.
    """
    import numpy as np
    rhos = []
    for col in returns_df.columns:
        s = returns_df[col].to_numpy(dtype=float)
        s = s[~np.isnan(s)]
        if len(s) < 3 or np.std(s[:-1]) == 0 or np.std(s[1:]) == 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = np.corrcoef(s[:-1], s[1:])[0, 1]
        if np.isfinite(rho):
            rhos.append(rho)
    if not rhos:
        return 1.0
    rho_med = float(np.median(rhos))
    rho_med = min(max(rho_med, 0.0), 0.95)         # clamp to keep b finite/sane
    return max(1.0, round(1.0 / (1.0 - rho_med)))
