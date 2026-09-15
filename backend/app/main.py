import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import benchmark, compare, query
from core.faiss_index import FaissHNSW
from core.hnsw import HNSW

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
BENCHMARK_RESULTS_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "benchmarks" / "results" / "comparison_full100k.json"
)


def _load_id_to_text(metadata_path):
    id_to_text = {}
    with open(metadata_path) as f:
        for line in f:
            row = json.loads(line)
            id_to_text[row["id"]] = row["text"]
    return id_to_text


def _check_ids_covered(index_name, index_ids, id_to_text):
    """Fails loudly at startup, not per-request, if an index's ids aren't
    a subset of id_to_text's keys.

    Found during an exhaustive review: core/rag.py's retrieve() does a raw
    id_to_text[node_id] lookup with no fallback, so a persisted index
    (data/hnsw_index.pkl or faiss_index.bin) built from a different corpus
    snapshot than the currently-loaded data/corpus_metadata.jsonl would
    raise an unhandled KeyError -> 500 on whichever query happens to
    retrieve the mismatched id -- unpredictable, and easy to miss until a
    demo hits it live. scripts/build_and_save_indices.py already asserts
    len(ids) == len(embeddings) for the same class of drift at build time;
    this is the equivalent check for the two artifacts that are actually
    loaded together at serve time.
    """
    missing = set(index_ids) - id_to_text.keys()
    if missing:
        sample = sorted(missing)[:5]
        raise RuntimeError(
            f"{index_name} contains {len(missing)} id(s) not present in "
            f"corpus_metadata.jsonl (e.g. {sample}) -- the persisted index "
            f"and the metadata file were built from different corpus "
            f"snapshots. Rerun scripts/build_and_save_indices.py so both "
            f"are built from the same data/corpus_embeddings.npy."
        )


def create_app(hnsw_index=None, faiss_index=None, id_to_text=None, benchmark_results=None):
    """Factory, not a bare module-level app -- lets tests (Batch 8's
    test_api.py) inject small, fast-to-build real indices instead of
    forcing every test run through the ~5-6 minute full-corpus build.
    Any argument left as None falls back to loading the real, persisted
    100k-corpus artifacts (scripts/build_and_save_indices.py) -- the path
    the actual running server takes.
    """
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.hnsw_index = (
        hnsw_index if hnsw_index is not None
        else HNSW.load(DATA_DIR / "hnsw_index.pkl")
    )
    app.state.faiss_index = (
        faiss_index if faiss_index is not None
        else FaissHNSW.load(DATA_DIR / "faiss_index.bin", DATA_DIR / "faiss_ids.pkl")
    )
    app.state.id_to_text = (
        id_to_text if id_to_text is not None
        else _load_id_to_text(DATA_DIR / "corpus_metadata.jsonl")
    )
    app.state.benchmark_results = (
        benchmark_results if benchmark_results is not None
        else json.loads(BENCHMARK_RESULTS_PATH.read_text())
    )

    _check_ids_covered("hnsw_index", app.state.hnsw_index.vectors.keys(), app.state.id_to_text)
    _check_ids_covered("faiss_index", app.state.faiss_index._ids, app.state.id_to_text)

    # /api prefix mirrors BuildBoard's convention: keeps API resource paths
    # distinct from whatever client-side routes the frontend (Batch 9) adds
    # once both are served from the same domain.
    app.include_router(query.router, prefix="/api")
    app.include_router(compare.router, prefix="/api")
    app.include_router(benchmark.router, prefix="/api")

    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    return app


# No module-level `app = create_app()` here on purpose: that would run
# create_app()'s real disk-loading fallback (390MB HNSW pickle + FAISS
# index) the instant this module is merely IMPORTED -- including by
# test_api.py, which only wants the create_app function itself to build a
# small test app with injected fixtures. Run the real server with
# uvicorn's factory mode instead, which calls create_app() lazily:
#     uvicorn app.main:create_app --factory --app-dir backend
