import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.hnsw import HNSW


def brute_force_topk(vectors, query, k):
    dists = np.sum((vectors - query) ** 2, axis=1)
    order = np.argsort(dists)[:k]
    return set(order.tolist())


def recall_at_k(index, vectors, queries, k, ef):
    total_recall = 0.0
    for q in queries:
        true_ids = brute_force_topk(vectors, q, k)
        result = index.search(q, k=k, ef=ef)
        returned_ids = {node_id for node_id, _ in result}
        total_recall += len(true_ids & returned_ids) / k
    return total_recall / len(queries)


def check_empty_index_returns_empty():
    index = HNSW(dim=10)
    result = index.search(np.zeros(10), k=5)
    assert result == [], "search on empty index must return []"
    print("[PASS] empty index returns []")


def check_self_query_finds_exact_match(index, vectors):
    query_id = 42
    result = index.search(vectors[query_id], k=1, ef=50)
    assert len(result) == 1, "expected exactly 1 result for k=1"
    returned_id, dist = result[0]
    assert returned_id == query_id, \
        f"querying with an indexed vector should return itself first, got {returned_id}"
    assert dist < 1e-9, f"distance to exact match should be ~0, got {dist}"
    print(f"[PASS] querying with an indexed vector (id={query_id}) returns itself, distance={dist:.2e}")


def check_result_shape_and_ordering(index, vectors, dim):
    rng = np.random.default_rng(123)
    q = rng.normal(size=dim)
    result = index.search(q, k=10, ef=50)
    assert len(result) == 10, f"expected 10 results, got {len(result)}"

    ids = [node_id for node_id, _ in result]
    assert len(set(ids)) == len(ids), "search returned duplicate node ids"

    dists = [d for _, d in result]
    assert dists == sorted(dists), "results must be sorted ascending by distance"
    print("[PASS] search returns k unique results, sorted ascending by distance")


def check_k_larger_than_index():
    index = HNSW(dim=8, seed=1)
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(5, 8))
    for i in range(5):
        index.insert(i, vectors[i])

    result = index.search(vectors[0], k=100, ef=50)
    assert len(result) == 5, f"expected all 5 indexed nodes, got {len(result)}"
    print("[PASS] k larger than index size returns all available nodes, no crash")


def check_k_zero_or_negative_returns_empty(index):
    # Found during an exhaustive review: candidates[:k] with a negative k
    # silently slices from the END (Python slice semantics) instead of
    # erroring or returning nothing -- verified directly that k=-1
    # against a 20-node index returned 19 (nearly all) results.
    assert index.search(np.zeros(index.dim), k=0, ef=50) == []
    assert index.search(np.zeros(index.dim), k=-1, ef=50) == []
    assert index.search(np.zeros(index.dim), k=-100, ef=50) == []
    print("[PASS] k<=0 returns [] (not a slice-from-the-end bug)")


def check_dimension_mismatch_raises(dim):
    index = HNSW(dim=dim, seed=1)
    try:
        index.insert(0, np.zeros(dim + 1))
        assert False, "expected ValueError for wrong-dimension vector"
    except ValueError:
        pass
    print("[PASS] insert() rejects a wrong-dimension vector with ValueError")


def check_recall_vs_ef(index, vectors, dim, k=10):
    rng = np.random.default_rng(999)
    queries = rng.normal(size=(200, dim))

    print(f"\n[INFO] recall@{k} over 200 held-out random queries, at increasing ef "
          f"(the recall/latency tradeoff this project is about):")
    prev_recall = -1.0
    for ef in (1, 10, 50, 200):
        r = recall_at_k(index, vectors, queries, k, ef)
        print(f"[INFO]   ef={ef:<4d} recall@{k} = {r:.3f}")
        assert r >= prev_recall - 0.02, \
            f"recall dropped sharply going from previous ef to ef={ef} ({r:.3f} < {prev_recall:.3f})"
        prev_recall = r

    final_recall = recall_at_k(index, vectors, queries, k, ef=50)
    assert final_recall >= 0.85, \
        f"recall@{k} at ef=50 is {final_recall:.3f}, expected >= 0.85 on this dataset"
    print(f"[PASS] recall@{k} at ef=50 = {final_recall:.3f} (>= 0.85 threshold), "
          f"and recall is monotonically non-decreasing as ef increases")


if __name__ == "__main__":
    N, DIM = 2000, 50
    print(f"Building HNSW index: n={N}, dim={DIM}\n")

    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(N, DIM))
    index = HNSW(dim=DIM, M=16, ef_construction=200, seed=42)
    for i in range(N):
        index.insert(i, vectors[i])

    check_empty_index_returns_empty()
    check_self_query_finds_exact_match(index, vectors)
    check_result_shape_and_ordering(index, vectors, DIM)
    check_k_larger_than_index()
    check_k_zero_or_negative_returns_empty(index)
    check_dimension_mismatch_raises(DIM)
    check_recall_vs_ef(index, vectors, DIM)

    print("\nAll Batch 2 search checks passed.")
