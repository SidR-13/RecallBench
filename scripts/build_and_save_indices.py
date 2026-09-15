"""Build both indices ONCE on the full real corpus and persist them to
disk, so the FastAPI server (Batch 8) can load a ready-built index in
seconds on startup instead of repeating a ~3.5 minute (HNSW) / ~2 minute
(FAISS) build on every restart (see Batch 6's measured build times).

Indexes the FULL corpus (no held-out queries) -- unlike
scripts/run_benchmark.py, this isn't measuring recall against a held-out
split, it's building the real index a live demo actually searches, so
every real paragraph should be searchable.

Run once (or whenever data/corpus_embeddings.npy changes):
    python scripts/build_and_save_indices.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.corpus import load_saved_corpus
from core.faiss_index import FaissHNSW
from core.hnsw import HNSW

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_PATH = DATA_DIR / "corpus_embeddings.npy"
METADATA_PATH = DATA_DIR / "corpus_metadata.jsonl"

HNSW_INDEX_PATH = DATA_DIR / "hnsw_index.pkl"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
FAISS_IDS_PATH = DATA_DIR / "faiss_ids.pkl"


def main():
    print("Loading full corpus...")
    ids, embeddings = load_saved_corpus(EMBEDDINGS_PATH, METADATA_PATH)
    dim = embeddings.shape[1]
    print(f"  {len(ids)} vectors, dim={dim}")

    print("\nBuilding HNSW (M=16, ef_construction=200, seed=42)...")
    t0 = time.perf_counter()
    hnsw = HNSW(dim=dim, M=16, ef_construction=200, seed=42)
    for node_id, vec in zip(ids, embeddings):
        hnsw.insert(node_id, vec)
    print(f"  build time: {time.perf_counter() - t0:.1f}s")
    hnsw.save(HNSW_INDEX_PATH)
    print(f"  saved to {HNSW_INDEX_PATH}")

    print("\nBuilding FAISS (M=16, ef_construction=200)...")
    t0 = time.perf_counter()
    faiss_index = FaissHNSW(dim=dim, M=16, ef_construction=200)
    for node_id, vec in zip(ids, embeddings):
        faiss_index.insert(node_id, vec)
    print(f"  build time: {time.perf_counter() - t0:.1f}s")
    faiss_index.save(FAISS_INDEX_PATH, FAISS_IDS_PATH)
    print(f"  saved to {FAISS_INDEX_PATH} + {FAISS_IDS_PATH}")


if __name__ == "__main__":
    main()
