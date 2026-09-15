import json
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.main import create_app
from core.config import settings
from core.faiss_index import FaissHNSW
from core.hnsw import HNSW

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
N_CORPUS = 3000  # same small-real-subset convention as test_rag.py / test_cross_index_retrieval.py


def build_test_state(n=N_CORPUS, seed=42):
    embeddings = np.load(DATA_DIR / "corpus_embeddings.npy")
    ids = []
    with open(DATA_DIR / "corpus_metadata.jsonl") as f:
        for line in f:
            ids.append(json.loads(line)["id"])
    ids = np.asarray(ids)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ids), size=n, replace=False)
    embeddings, ids = embeddings[idx], ids[idx]

    wanted = set(ids.tolist())
    id_to_text = {}
    with open(DATA_DIR / "corpus_metadata.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] in wanted:
                id_to_text[row["id"]] = row["text"]

    dim = embeddings.shape[1]
    hnsw = HNSW(dim=dim, M=16, ef_construction=200, seed=42)
    faiss_index = FaissHNSW(dim=dim, M=16, ef_construction=200)
    for node_id, vec in zip(ids, embeddings):
        hnsw.insert(node_id, vec)
        faiss_index.insert(node_id, vec)

    fake_benchmark_results = {"n": n, "note": "fake results injected for test_api.py"}
    return hnsw, faiss_index, id_to_text, fake_benchmark_results


def make_client():
    hnsw, faiss_index, id_to_text, benchmark_results = build_test_state()
    app = create_app(
        hnsw_index=hnsw,
        faiss_index=faiss_index,
        id_to_text=id_to_text,
        benchmark_results=benchmark_results,
    )
    return TestClient(app), benchmark_results


def check_health():
    client, _ = make_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("[PASS] GET /health returns 200 {'status': 'ok'}")


def check_query_endpoint_real_retrieval_mock_answer():
    # Forces ai_mock True for this check's duration: it verifies retrieval
    # + the mock-answer contract deterministically and for free, regardless
    # of whether this environment's .env has a real ANTHROPIC_API_KEY. A
    # real (billed) call per test run would be the wrong default for a
    # suite meant to run repeatedly and in CI.
    original = settings.ai_mock
    settings.ai_mock = True
    try:
        client, _ = make_client()
        response = client.post(
            "/api/query",
            json={"query": "What is the greenhouse effect?", "k": 3, "ef": 100},
        )
        assert response.status_code == 200
        body = response.json()

        assert body["query"] == "What is the greenhouse effect?"
        assert len(body["retrieved"]) == 3
        for chunk in body["retrieved"]:
            assert len(chunk["text"].strip()) > 0
        assert body["answer"].startswith("[MOCK]")
    finally:
        settings.ai_mock = original

    print(f"\n[INFO] POST /api/query retrieved real text (first hit): "
          f"{body['retrieved'][0]['text'][:120].strip()!r}...")
    print("[PASS] /api/query returns real retrieved text and a labeled mock answer")


def check_compare_endpoint_both_indices_real_text():
    client, _ = make_client()
    response = client.post(
        "/api/compare",
        json={"query": "history of ancient Rome", "k": 5, "ef": 100},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["query"] == "history of ancient Rome"
    assert len(body["our_hnsw"]) == 5
    assert len(body["faiss"]) == 5

    hnsw_ids = {chunk["id"] for chunk in body["our_hnsw"]}
    faiss_ids = {chunk["id"] for chunk in body["faiss"]}
    overlap = len(hnsw_ids & faiss_ids)

    for chunk in body["our_hnsw"] + body["faiss"]:
        assert len(chunk["text"].strip()) > 0

    print(f"\n[INFO] POST /api/compare: our_hnsw/faiss top-5 overlap = {overlap}/5")
    print(f"[INFO]   our_hnsw[0]: {body['our_hnsw'][0]['text'][:100].strip()!r}...")
    print(f"[INFO]   faiss[0]:    {body['faiss'][0]['text'][:100].strip()!r}...")
    assert overlap >= 3, f"expected substantial agreement, got {overlap}/5"
    print("[PASS] /api/compare returns real text from both indices with substantial agreement")


def check_invalid_k_ef_rejected_with_422():
    # Found during an exhaustive review: k=0 crashed FaissHNSW.search()
    # with a raw, uncaught FAISS C++ AssertionError -> unhandled 500;
    # negative k silently returned wrong results from HNSW's Python slice
    # semantics; and neither k nor ef had an upper bound, letting one
    # client occupy a thread-pool worker with an arbitrarily expensive
    # search. RetrievalRequest (backend/app/schemas.py) now rejects all of
    # these at the network boundary with a clean 422, before any of that
    # code runs -- verified here against both /api/query and /api/compare,
    # since both share the same request model.
    client, _ = make_client()
    bad_bodies = [
        {"query": "test", "k": 0, "ef": 100},
        {"query": "test", "k": -1, "ef": 100},
        {"query": "test", "k": 5, "ef": 0},
        {"query": "test", "k": 5, "ef": -10},
        {"query": "test", "k": 100_000, "ef": 100},  # over the k bound
        {"query": "test", "k": 5, "ef": 100_000},  # over the ef bound
    ]
    for path in ("/api/query", "/api/compare"):
        for body in bad_bodies:
            response = client.post(path, json=body)
            assert response.status_code == 422, (
                f"{path} with {body} expected 422, got {response.status_code}: "
                f"{response.text[:200]}"
            )
    print("[PASS] invalid k/ef (<=0, over bounds) rejected with 422 on both "
          "/api/query and /api/compare, before reaching index code")


def check_startup_rejects_index_metadata_mismatch():
    # Found during an exhaustive review: core/rag.py's retrieve() does a
    # raw id_to_text[node_id] lookup with no fallback -- if a persisted
    # index and corpus_metadata.jsonl were ever built from different
    # corpus snapshots, this would surface as an unpredictable, unhandled
    # KeyError -> 500 on whichever query happens to touch the mismatched
    # id. create_app() now validates this at startup instead, matching
    # scripts/build_and_save_indices.py's own len(ids)==len(embeddings)
    # check for the same class of drift.
    hnsw, faiss_index, id_to_text, benchmark_results = build_test_state()

    # Corrupt id_to_text so it's missing an id the indices actually contain.
    some_id = next(iter(id_to_text))
    broken_id_to_text = {k: v for k, v in id_to_text.items() if k != some_id}

    try:
        create_app(
            hnsw_index=hnsw,
            faiss_index=faiss_index,
            id_to_text=broken_id_to_text,
            benchmark_results=benchmark_results,
        )
        assert False, "expected RuntimeError for index/metadata id mismatch"
    except RuntimeError as e:
        assert "corpus_metadata" in str(e) or "id(s)" in str(e)
    print("[PASS] create_app() fails loudly at startup on index/metadata id "
          "mismatch, instead of a per-request KeyError later")


def check_benchmark_endpoint_serves_injected_results():
    client, expected_results = make_client()
    response = client.get("/api/benchmark")
    assert response.status_code == 200
    assert response.json() == expected_results
    print("[PASS] GET /api/benchmark serves the precomputed results as is, no live re-run")


if __name__ == "__main__":
    check_health()
    check_query_endpoint_real_retrieval_mock_answer()
    check_compare_endpoint_both_indices_real_text()
    check_invalid_k_ef_rejected_with_422()
    check_startup_rejects_index_metadata_mismatch()
    check_benchmark_endpoint_serves_injected_results()
    print("\nAll Batch 8 API checks passed.")
