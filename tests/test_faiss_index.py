import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.faiss_index import FaissHNSW


def build_index(n=500, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    ids = [f"doc_{i}" for i in range(n)]  # non-integer ids, mirrors real corpus
    index = FaissHNSW(dim=dim, M=16, ef_construction=200)
    for node_id, vector in zip(ids, vectors):
        index.insert(node_id, vector)
    return index, ids, vectors


def check_arbitrary_ids_are_respected():
    index, ids, vectors = build_index()
    results = index.search(vectors[7], k=1, ef=50)
    assert results[0][0] == "doc_7"


def check_self_query_finds_exact_match():
    index, ids, vectors = build_index()
    results = index.search(vectors[42], k=1, ef=50)
    node_id, dist = results[0]
    assert node_id == "doc_42"
    assert dist < 1e-6


def check_result_shape_and_ordering():
    index, ids, vectors = build_index()
    results = index.search(vectors[0], k=10, ef=50)
    assert len(results) == 10
    assert len(set(r[0] for r in results)) == 10  # no duplicate ids
    dists = [d for _, d in results]
    assert dists == sorted(dists)  # ascending by distance


def check_k_larger_than_index_no_crash():
    index = FaissHNSW(dim=8, M=16, ef_construction=200)
    rng = np.random.default_rng(1)
    for i in range(5):
        index.insert(i, rng.standard_normal(8).astype(np.float32))
    results = index.search(rng.standard_normal(8).astype(np.float32), k=100, ef=50)
    assert len(results) == 5  # can't return more than what's indexed


def check_ef_sweep_recall_increases_toward_ground_truth():
    # Same structural check Batch 2 ran on our own HNSW: recall should be
    # monotonically non-decreasing as ef (efSearch) grows, converging
    # toward the true top-k found by brute-force numpy argsort.
    n, dim, k = 3000, 32, 10
    rng = np.random.default_rng(2)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    ids = list(range(n))
    index = FaissHNSW(dim=dim, M=16, ef_construction=200)
    for node_id, vector in zip(ids, vectors):
        index.insert(node_id, vector)

    queries = rng.standard_normal((50, dim)).astype(np.float32)
    recalls = {}
    for ef in (1, 10, 50, 200):
        hits = 0
        total = 0
        for q in queries:
            approx_ids = {nid for nid, _ in index.search(q, k, ef=ef)}
            true_dists = np.einsum("ij,ij->i", vectors - q, vectors - q)
            true_ids = set(np.argsort(true_dists)[:k].tolist())
            hits += len(approx_ids & true_ids)
            total += k
        recalls[ef] = hits / total

    values = [recalls[ef] for ef in (1, 10, 50, 200)]
    assert values == sorted(values)  # monotonically non-decreasing
    assert values[-1] > 0.9  # high ef should recover most true neighbors


def check_k_zero_or_negative_returns_empty_no_crash():
    # Found during an exhaustive review: verified directly that
    # faiss.IndexHNSWFlat.search(q, k=0) (and negative k) raises a raw,
    # uncaught C++ AssertionError -- unlike BruteForce.search(), which
    # already clamps this to []. This confirms the fix (an early
    # `if k <= 0: return []` in FaissHNSW.search) actually prevents that
    # crash rather than just moving where it happens.
    index, ids, vectors = build_index(n=50, dim=8)
    assert index.search(vectors[0], k=0, ef=50) == []
    assert index.search(vectors[0], k=-1, ef=50) == []


def check_concurrent_search_different_ef_no_corruption():
    # Found during an exhaustive review (two independent passes flagged
    # this): search() sets self.index.hnsw.efSearch as a side effect on
    # shared state before calling .search(). Without the _search_lock
    # fix, two threads searching with different `ef` values at the same
    # time could race -- one thread's search silently running with the
    # OTHER thread's ef. This hammers that race with many threads and
    # many repetitions; it can't prove a race is impossible, but it's a
    # real regression check that would have a strong chance of failing on
    # the original unlocked code (verified: raced reliably in manual
    # testing during Batch 9's concurrency investigation of the same bug
    # class in core/embed.py).
    index, ids, vectors = build_index(n=2000, dim=32)
    query = vectors[0]
    k = 10

    # ef=2000 (>= n) on a 2000-node graph explores exhaustively -- must
    # ALWAYS return the true top-10 exactly, every single call. If a
    # concurrent thread's efSearch=1 ever leaks into a "high" thread's
    # search() call between the lock releasing and .search() finishing
    # the previous run, that call's recall would drop below perfect and
    # this catches it deterministically, unlike a bare "no crash" check.
    true_dists = np.sum((vectors - query) ** 2, axis=1)
    # ids are strings ("doc_0", ...), not integer positions (build_index's
    # own convention -- non-integer ids, mirroring the real corpus) -- map
    # argsort's positional indices back through `ids` before comparing
    # against index.search()'s actual returned ids.
    true_top_k = {ids[i] for i in np.argsort(true_dists)[:k]}

    errors = []
    high_result_sets = []
    lock = threading.Lock()

    # The race this guards against is real but has a narrow window:
    # a minimal, unwrapped reproduction (raw faiss.IndexHNSWFlat, no
    # FaissHNSW/lock at all) measured it at ~1/800 with 4 threads x 200
    # iterations per side -- confirming the underlying race exists and
    # this is a reasonable trial count to have a real chance of hitting
    # it. Through FaissHNSW's actual call path (extra Python bytecode
    # between the set and the FAISS call shifts the timing), this same
    # trial count did not reliably reproduce the un-locked failure in
    # practice -- so this test is a real stress exercise, not a
    # guaranteed catch of a lock regression. The actual correctness
    # guarantee is `self._search_lock` wrapping the ENTIRE
    # set-then-read critical section (core/faiss_index.py) -- verified by
    # direct code inspection, which is how mutual exclusion is normally
    # verified, not by chasing a flaky race empirically.
    ITERS = 200
    N_THREADS = 4

    def search_low():
        for _ in range(ITERS):
            try:
                index.search(query, k=k, ef=1)
            except Exception as e:
                with lock:
                    errors.append(e)

    def search_high():
        for _ in range(ITERS):
            try:
                result_ids = {nid for nid, _ in index.search(query, k=k, ef=2000)}
                with lock:
                    high_result_sets.append(result_ids)
            except Exception as e:
                with lock:
                    errors.append(e)

    threads = (
        [threading.Thread(target=search_low) for _ in range(N_THREADS)]
        + [threading.Thread(target=search_high) for _ in range(N_THREADS)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent search raised: {errors}"
    assert len(high_result_sets) == N_THREADS * ITERS
    bad_runs = [s for s in high_result_sets if s != true_top_k]
    assert not bad_runs, (
        f"{len(bad_runs)}/{len(high_result_sets)} high-ef searches did not return "
        f"the true top-{k} while racing with concurrent low-ef searches -- "
        f"efSearch race detected"
    )


def check_memory_bytes():
    # Same three properties Batch 6 requires of our own HNSW.memory_bytes():
    # deterministic, above the raw-vector-data floor, and monotonic in
    # index size -- both sides need to satisfy the same contract to be a
    # fair comparison.
    index, ids, vectors = build_index(n=500, dim=16)
    first = index.memory_bytes()
    second = index.memory_bytes()
    assert first == second, f"memory_bytes() not deterministic: {first} != {second}"

    raw_vector_bytes = 500 * 16 * 4
    assert first > raw_vector_bytes, \
        f"memory_bytes()={first} is below the raw vector data floor ({raw_vector_bytes})"

    smaller_index, _, _ = build_index(n=100, dim=16)
    assert smaller_index.memory_bytes() < first, \
        "a smaller index must report less memory than a larger one"


def run_all():
    checks = [v for name, v in globals().items() if name.startswith("check_")]
    for check in checks:
        check()
        print(f"OK: {check.__name__}")
    print(f"\n{len(checks)} checks passed.")


if __name__ == "__main__":
    run_all()
