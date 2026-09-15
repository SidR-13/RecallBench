"""Benchmark measurement primitives, shared by both the HNSW and (Batch 6)
FAISS sides of the comparison. Kept independent of any specific index type —
every function here takes plain data in and returns a plain number out, so
neither side of the benchmark gets special-cased measurement logic.
"""

import time

import numpy as np


def time_build(insert_fn, ids, vectors):
    """Time how long it takes to insert every (id, vector) pair.

    `insert_fn` is a callable insert_fn(node_id, vector) -- not the index
    object itself -- so this works for any index type (HNSW.insert or, in
    Batch 6, whatever FAISS's add method is wrapped as) without this
    function needing to know which one it's timing.

    Returns build time in seconds.
    """
    start = time.perf_counter()
    for node_id, vector in zip(ids, vectors):
        insert_fn(node_id, vector)
    return time.perf_counter() - start


def recall_at_k(approx_ids, exact_ids):
    """Recall@k for one query: the fraction of the true top-k (`exact_ids`,
    from BruteForce) that also appear in the approximate top-k
    (`approx_ids`, from HNSW or FAISS). Order doesn't matter for this
    metric -- only whether each true neighbor was found at all.
    """
    if len(exact_ids) == 0:
        return 1.0  # nothing to find, vacuously perfect -- avoids 0/0
    approx_set = set(approx_ids)
    hits = sum(1 for eid in exact_ids if eid in approx_set)
    return hits / len(exact_ids)


def latency_percentiles(latencies_seconds, percentiles=(50, 95)):
    """Convert a list of per-query wall-clock times (seconds) into the
    requested percentiles, reported in milliseconds (the unit that reads
    naturally in a resume/README table).
    """
    arr = np.asarray(latencies_seconds, dtype=np.float64) * 1000.0
    return {p: float(np.percentile(arr, p)) for p in percentiles}
