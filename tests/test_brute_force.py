import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.brute_force import BruteForce
from core.hnsw import HNSW


def check_hand_checkable_example():
    # Points on a line: 0, 1, 2, 5, 10. Query at 3.
    # Distances from 3 (squared): 9, 4, 1, 4, 49
    # True order by distance: 2 (d=1), then 1 and 5 tie (d=4), then 0 (d=9), then 10 (d=49)
    vectors = np.array([[0.0], [1.0], [2.0], [5.0], [10.0]])
    bf = BruteForce(vectors)
    result = bf.search(np.array([3.0]), k=3)

    ids = [node_id for node_id, _ in result]
    dists = [d for _, d in result]

    assert ids[0] == 2 and dists[0] == 1.0, f"nearest to 3 should be point 2 (d=1), got {result[0]}"
    assert set(ids[1:3]) == {1, 3}, f"2nd/3rd nearest should be points 1 and 3 (both d=4), got {ids[1:3]}"
    assert dists[1] == 4.0 and dists[2] == 4.0
    print(f"[PASS] hand-checkable 1D example: query=3.0 -> {result} matches computed-by-hand distances")


def check_ids_are_respected():
    vectors = np.array([[0.0, 0.0], [10.0, 10.0], [1.0, 1.0]])
    custom_ids = ["doc_a", "doc_b", "doc_c"]
    bf = BruteForce(vectors, ids=custom_ids)
    result = bf.search(np.array([0.9, 0.9]), k=1)
    assert result[0][0] == "doc_c", f"expected doc_c (closest to [0.9,0.9]), got {result[0][0]}"
    print(f"[PASS] custom ids respected and returned: {result}")


def check_k_larger_than_corpus():
    vectors = np.random.default_rng(0).normal(size=(4, 8))
    bf = BruteForce(vectors)
    result = bf.search(vectors[0], k=100)
    assert len(result) == 4, f"expected all 4 vectors, got {len(result)}"
    print("[PASS] k larger than corpus returns all available vectors, no crash")


def check_return_types_are_json_safe():
    vectors = np.random.default_rng(0).normal(size=(10, 4))
    bf = BruteForce(vectors)
    result = bf.search(vectors[0], k=3)
    for node_id, dist in result:
        assert isinstance(node_id, int), f"node_id should be python int, got {type(node_id)}"
        assert isinstance(dist, float), f"distance should be python float, got {type(dist)}"
    print("[PASS] returned ids/distances are plain python int/float (JSON-serializable)")


def check_results_sorted_ascending():
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(500, 20))
    bf = BruteForce(vectors)
    result = bf.search(rng.normal(size=20), k=50)
    dists = [d for _, d in result]
    assert dists == sorted(dists), "results must be sorted ascending by distance"
    print("[PASS] results sorted ascending by distance")


def check_agrees_with_independent_numpy_computation():
    # Recompute ground truth a completely different way (full sort, no
    # argpartition) and confirm BruteForce.search produces the identical
    # id set and identical distances -- this is the regression check that
    # BruteForce itself isn't the thing introducing an error.
    rng = np.random.default_rng(11)
    n, dim, k = 3000, 30, 15
    vectors = rng.normal(size=(n, dim))
    query = rng.normal(size=dim)

    bf = BruteForce(vectors)
    result = bf.search(query, k=k)

    dists_full = np.sum((vectors - query) ** 2, axis=1)
    independent_order = np.argsort(dists_full)[:k]
    independent_ids = set(independent_order.tolist())

    returned_ids = {node_id for node_id, _ in result}
    assert returned_ids == independent_ids, \
        "BruteForce.search disagrees with an independently computed full argsort"

    # BruteForce stores vectors as float32 (see Batch 4 decision: matches
    # native embedding-model precision, keeps the memory benchmark honest),
    # while `dists_full` here is computed in float64 -- so exact equality
    # isn't the right check anymore. float32 carries ~7 significant decimal
    # digits, so a relative tolerance is the mathematically correct
    # comparison (fixed absolute epsilons don't scale with magnitude).
    for (node_id, dist), idx in zip(result, independent_order):
        assert node_id == idx
        assert np.isclose(dist, dists_full[idx], rtol=1e-5, atol=1e-4), \
            f"distance mismatch beyond float32 precision: {dist} vs {dists_full[idx]}"
    print(f"[PASS] BruteForce.search matches an independently computed full-argsort "
          f"ground truth within float32 precision, on {n} vectors, dim={dim}, k={k}")


def check_consistency_with_hnsw_distance_metric():
    # Confirm BruteForce and HNSW compute the literal same distance for the
    # same pair of vectors -- if these ever diverged, recall numbers
    # computed against BruteForce would be meaningless.
    rng = np.random.default_rng(3)
    a = rng.normal(size=16)
    b = rng.normal(size=16)

    index = HNSW(dim=16)
    index.insert(0, a)
    # _distance_to_query's precondition is that `query` is already float32
    # -- normally guaranteed by its only real callers, search()/insert(),
    # which cast before calling down. Calling it directly means honoring
    # that precondition ourselves, same as search() would.
    hnsw_dist = index._distance_to_query(b.astype(np.float32), 0)

    bf = BruteForce(np.array([a]))
    bf_dist = bf.search(b, k=1)[0][1]

    assert abs(hnsw_dist - bf_dist) < 1e-9, \
        f"HNSW and BruteForce disagree on distance: {hnsw_dist} vs {bf_dist}"
    print(f"[PASS] HNSW and BruteForce agree on squared-L2 distance for the same pair "
          f"({hnsw_dist:.6f} vs {bf_dist:.6f})")


def check_k_zero_returns_empty():
    vectors = np.random.default_rng(0).normal(size=(20, 5))
    bf = BruteForce(vectors)
    result = bf.search(vectors[0], k=0)
    assert result == [], f"k=0 should return [], got {result}"
    print("[PASS] k=0 returns [], no crash")


def check_tie_breaking_is_deterministic():
    # Four points, two exact pairs of ties in distance from the query.
    # point 0 and point 1 are both distance 1 from the query.
    # point 2 and point 3 are both distance 4 from the query.
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    query = np.array([0.0, 0.0])
    ids = [10, 5, 30, 20]  # deliberately out of order, to prove id-based tie-break isn't accidental
    bf = BruteForce(vectors, ids=ids)

    result = bf.search(query, k=4)
    dists = [d for _, d in result]
    assert dists == [1.0, 1.0, 4.0, 4.0]

    # Within each tied group, ties must break by ascending id: (5 before 10), (20 before 30).
    tied_group_1 = [result[0][0], result[1][0]]
    tied_group_2 = [result[2][0], result[3][0]]
    assert tied_group_1 == [5, 10], f"tie group 1 should break by ascending id [5, 10], got {tied_group_1}"
    assert tied_group_2 == [20, 30], f"tie group 2 should break by ascending id [20, 30], got {tied_group_2}"

    # Run again to confirm this isn't a one-off — must be reproducible every call.
    result_again = bf.search(query, k=4)
    assert result == result_again, "search() must return identical results across repeated calls on the same input"
    print(f"[PASS] tied distances break deterministically by ascending id, and are reproducible across repeated calls: {result}")


def check_mismatched_ids_length_raises():
    # Found during an exhaustive review: BruteForce is the ground-truth
    # oracle every recall number in this project is measured against, but
    # had no check that ids and vectors were even the same length -- a
    # caller-side off-by-one would previously either raise a confusing
    # IndexError far from the real cause, or (if ids were longer) silently
    # return a real-looking but wrong id.
    vectors = np.zeros((5, 3))
    try:
        BruteForce(vectors, ids=[1, 2, 3])  # too few ids for 5 vectors
        assert False, "expected ValueError for mismatched ids/vectors length"
    except ValueError:
        pass
    print("[PASS] mismatched ids/vectors length raises ValueError at construction")


if __name__ == "__main__":
    check_hand_checkable_example()
    check_ids_are_respected()
    check_k_larger_than_corpus()
    check_return_types_are_json_safe()
    check_results_sorted_ascending()
    check_agrees_with_independent_numpy_computation()
    check_consistency_with_hnsw_distance_metric()
    check_k_zero_returns_empty()
    check_tie_breaking_is_deterministic()
    check_mismatched_ids_length_raises()
    print("\nAll Batch 3 brute-force checks passed.")
