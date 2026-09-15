import threading

import numpy as np
from sentence_transformers import SentenceTransformer

# Small, fast, free to run locally (no per-embedding API cost) -- chosen in
# CLAUDE_CONTEXT.md's tech stack. 384-dimensional output.
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
# Guards BOTH lazy model construction AND every encode() call below -- not
# just construction. As of Batch 8, FastAPI runs sync route handlers
# (query.py/compare.py) in a thread pool, so a single user action
# (Promise.all on the frontend) fires /api/query and /api/compare's
# embed_texts() calls from two threads at the same time. A lock around
# construction alone wasn't enough: concurrent first-time SentenceTransformer
# construction reliably crashed the entire server process (no Python
# traceback -- consistent with a native-level segfault in the
# PyTorch/Accelerate BLAS backend under concurrent initialization, not just
# a slow race) -- see CLAUDE_CONTEXT.md Batch 9 log. sentence-transformers'
# CPU inference isn't documented as safe to call from multiple threads at
# once either, so the lock covers the whole embed_texts() body, trading a
# small amount of parallelism (embedding calls queue instead of running
# side by side) for a server that cannot crash from this.
_model_lock = threading.Lock()


def get_model():
    # Loaded lazily and cached module-wide: the model only needs to be
    # read off disk once per process, not once per embed_texts() call.
    # Always called from inside _model_lock (see embed_texts) -- no
    # separate locking needed here.
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts, batch_size=128, show_progress=True):
    """Embed a list of strings with all-MiniLM-L6-v2.

    normalize_embeddings=True is the important flag here, not a detail:
    HNSW/BruteForce rank by squared L2, and squared-L2 ranking only equals
    cosine ranking when vectors are actually unit-length -- this is what
    makes that equivalence real rather than assumed (see Batch 1 log).

    Returns a (len(texts), 384) float32 array -- float32 to match the
    model's native output precision and keep the memory benchmark honest
    against FAISS (see Batch 4 log).
    """
    with _model_lock:
        model = get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
    return embeddings.astype(np.float32)
