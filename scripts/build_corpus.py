import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.corpus import load_wikipedia_paragraphs
from core.embed import embed_texts

N = 100_000
SEED = 42
PREFIX_ROWS = 150_000

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_PATH = OUT_DIR / "corpus_embeddings.npy"
METADATA_PATH = OUT_DIR / "corpus_metadata.jsonl"

if __name__ == "__main__":
    print(f"Loading {N} real Wikipedia paragraphs (seed={SEED}, prefix_rows={PREFIX_ROWS})...")
    t0 = time.time()
    ids, texts = load_wikipedia_paragraphs(n=N, seed=SEED, prefix_rows=PREFIX_ROWS)
    print(f"  loaded in {time.time() - t0:.1f}s")

    print(f"Embedding {N} paragraphs with all-MiniLM-L6-v2...")
    t0 = time.time()
    embeddings = embed_texts(texts, batch_size=128, show_progress=True)
    print(f"  embedded in {time.time() - t0:.1f}s -> shape {embeddings.shape}, dtype {embeddings.dtype}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(METADATA_PATH, "w") as f:
        for node_id, text in zip(ids, texts):
            f.write(json.dumps({"id": node_id, "text": text}) + "\n")

    size_mb = EMBEDDINGS_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved {EMBEDDINGS_PATH} ({size_mb:.1f} MB)")
    print(f"Saved {METADATA_PATH}")
    print(f"Corpus build complete: {N} real Wikipedia paragraphs, {embeddings.shape[1]}-dim embeddings.")
