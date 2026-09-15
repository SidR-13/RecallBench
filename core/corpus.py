import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# Real, pre-chunked (paragraph-level) Simple English Wikipedia text. This
# dataset also ships Cohere embeddings, which we deliberately ignore (see
# core/embed.py / CLAUDE_CONTEXT.md Batch 4 log) -- we only want its raw
# `contents` text, re-embedded ourselves with our own model for consistency
# with the rest of this project's pipeline.
DATASET_URL = (
    "https://huggingface.co/datasets/timescale/wikipedia-22-12-simple-embeddings"
    "/resolve/main/wiki.csv"
)

# From the dataset's own reported size: 4,472,602,158 bytes / 485,859 rows.
# Used to estimate how many bytes to download for a given row prefix,
# since the file's per-row size varies (the embedding column dominates
# row size and isn't fixed-width).
AVG_BYTES_PER_ROW = 9206

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _download_prefix(prefix_rows, cache_path, margin=1.3):
    """Download just enough of the source file's *byte prefix* to safely
    contain `prefix_rows` CSV rows, to a local file.

    pandas.read_csv(url) turned out to fetch over HTTP inefficiently row by
    row (measured: 5000 rows took 156s despite the underlying connection
    sustaining ~15MB/s for a plain byte-range request) -- downloading a
    bounded byte range with a real streaming client first, then having
    pandas parse the *local* file, is what actually makes use of the
    available bandwidth. `margin` over-fetches a bit since row size isn't
    fixed-width -- better to fetch slightly more than to fall short and
    silently get fewer rows than requested.
    """
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            existing_rows = sum(1 for _ in f) - 1  # minus header
        if existing_rows >= prefix_rows:
            return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    byte_count = int(prefix_rows * AVG_BYTES_PER_ROW * margin)

    subprocess.run(
        [
            "curl", "-sL",
            "--range", f"0-{byte_count}",
            "-o", str(cache_path),
            DATASET_URL,
        ],
        check=True,
    )


def load_wikipedia_paragraphs(n, seed=42, prefix_rows=200_000):
    """Load `n` real Wikipedia (Simple English) paragraphs.

    Downloads (and locally caches) only a `prefix_rows`-row byte-prefix of
    the ~485k-row source file rather than the whole ~4.4GB file, then takes
    a fixed-seed random sample of `n` rows from that prefix -- bounds
    download time and avoids the risk of a biased "first N rows" sample if
    row order correlates with anything (e.g. alphabetical-by-title).

    Returns (ids, texts): parallel lists, ids as plain python ints.
    """
    cache_path = CACHE_DIR / f"wiki_prefix_{prefix_rows}.csv"
    _download_prefix(prefix_rows, cache_path)

    df = pd.read_csv(cache_path, usecols=["id", "contents"], nrows=prefix_rows)

    if len(df) < prefix_rows:
        raise RuntimeError(
            f"expected {prefix_rows} rows after download, got {len(df)} -- "
            f"the byte-range download may have been too small; delete "
            f"{cache_path} and retry with a larger margin"
        )
    if n > len(df):
        raise ValueError(
            f"requested n={n} paragraphs but only {len(df)} available in the "
            f"{prefix_rows}-row prefix; increase prefix_rows or lower n"
        )

    sample = df.sample(n=n, random_state=seed).reset_index(drop=True)
    ids = sample["id"].astype(int).tolist()
    texts = sample["contents"].tolist()
    return ids, texts


def load_saved_corpus(embeddings_path, metadata_path):
    """Load the already-built (embeddings, ids) artifact pair produced by
    scripts/build_corpus.py -- the shared loader for anything that needs
    the real corpus after embedding, not the raw-Wikipedia loading above.

    Previously duplicated near-identically in both
    scripts/build_and_save_indices.py and scripts/run_benchmark.py (found
    during an exhaustive review) -- same np.load + per-line JSON scan +
    row-count check, copy-pasted rather than shared. Consolidating here
    means a fix to this logic (e.g. this file-existence check itself,
    added in the same review) only has to be made once.

    Raises real exceptions, not bare `assert` -- `python -O` (or
    PYTHONOPTIMIZE=1, which a CI/deploy pipeline could plausibly set)
    compiles out assert statements entirely, silently skipping this
    row-count safety check exactly when a corpus/metadata mismatch would
    otherwise be caught.
    """
    embeddings_path, metadata_path = Path(embeddings_path), Path(metadata_path)
    if not embeddings_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"corpus artifacts not found ({embeddings_path}, {metadata_path}) "
            f"-- run `python scripts/build_corpus.py` first"
        )

    embeddings = np.load(embeddings_path)
    ids = []
    with open(metadata_path) as f:
        for line in f:
            ids.append(json.loads(line)["id"])
    ids = np.asarray(ids)

    if len(ids) != len(embeddings):
        raise ValueError(
            f"metadata/embeddings row count mismatch: {len(ids)} ids in "
            f"{metadata_path} vs {len(embeddings)} rows in {embeddings_path} "
            f"-- these two files were built from different corpus runs; "
            f"regenerate both together via scripts/build_corpus.py"
        )

    return ids, embeddings
