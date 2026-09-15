import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.harness import (
    latency_percentiles,
    recall_at_k,
    time_build,
)


def check_recall_perfect_match():
    assert recall_at_k([1, 2, 3], [1, 2, 3]) == 1.0


def check_recall_partial_match():
    # 2 of the 4 true neighbors were found -> 0.5, regardless of order or
    # of extra wrong ids present in approx_ids.
    assert recall_at_k([1, 99, 2, 100], [1, 2, 3, 4]) == 0.5


def check_recall_no_match():
    assert recall_at_k([99, 100], [1, 2, 3]) == 0.0


def check_recall_empty_ground_truth():
    assert recall_at_k([1, 2, 3], []) == 1.0


def check_recall_order_independent():
    # Same sets, different order -- recall@k must not care about ranking,
    # only set membership (ranking quality is a separate concern this
    # metric doesn't claim to measure).
    assert recall_at_k([3, 1, 2], [1, 2, 3]) == recall_at_k([1, 2, 3], [1, 2, 3])


def check_latency_percentiles_known_values():
    # 100 evenly spaced latencies from 1ms to 100ms (as seconds) -> p50
    # should land near 50ms, p95 near 95ms. Hand-checkable, not a fuzzy
    # approximation.
    latencies = [i / 1000.0 for i in range(1, 101)]
    result = latency_percentiles(latencies, percentiles=(50, 95))
    assert abs(result[50] - 50.5) < 0.6
    assert abs(result[95] - 95.05) < 0.6


def check_latency_percentiles_single_value():
    result = latency_percentiles([0.01], percentiles=(50, 95))
    assert result[50] == 10.0
    assert result[95] == 10.0


def check_time_build_measures_real_duration():
    calls = []

    def fake_insert(node_id, vector):
        calls.append(node_id)
        time.sleep(0.01)

    ids = [1, 2, 3]
    vectors = [np.zeros(4), np.zeros(4), np.zeros(4)]
    elapsed = time_build(fake_insert, ids, vectors)

    assert calls == [1, 2, 3]  # every pair was actually inserted, in order
    assert elapsed >= 0.03  # 3 inserts x >=0.01s each, real elapsed time


def run_all():
    checks = [v for name, v in globals().items() if name.startswith("check_")]
    for check in checks:
        check()
        print(f"OK: {check.__name__}")
    print(f"\n{len(checks)} checks passed.")


if __name__ == "__main__":
    run_all()
