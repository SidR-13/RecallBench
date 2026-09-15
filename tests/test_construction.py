import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.hnsw import HNSW


def build_index(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim))
    index = HNSW(dim=dim, M=16, ef_construction=200, seed=42)
    for i in range(n):
        index.insert(i, vectors[i])
    return index, vectors


def check_layer0_completeness(index, n):
    assert set(index.graph[0].keys()) == set(range(n)), \
        "every inserted node must exist at layer 0"
    print(f"[PASS] all {n} nodes present at layer 0")


def check_bidirectional_edges(index):
    checked = 0
    for layer, adjacency in enumerate(index.graph):
        for node_id, neighbors in adjacency.items():
            for neighbor_id in neighbors:
                assert node_id in index.graph[layer][neighbor_id], \
                    f"edge {node_id}->{neighbor_id} at layer {layer} is one-directional"
                checked += 1
    print(f"[PASS] all edges bidirectional across all layers ({checked} directed edges checked)")


def check_degree_limits(index):
    for layer, adjacency in enumerate(index.graph):
        max_degree = index.M0 if layer == 0 else index.M
        for node_id, neighbors in adjacency.items():
            assert len(neighbors) <= max_degree, \
                f"node {node_id} at layer {layer} has degree {len(neighbors)} > {max_degree}"
    print("[PASS] no node exceeds its layer's max degree (M0 at layer 0, M above)")


def check_layer_membership_consistency(index, n):
    node_top_layer = {}
    for layer, adjacency in enumerate(index.graph):
        for node_id in adjacency:
            node_top_layer[node_id] = layer

    for node_id, top in node_top_layer.items():
        for layer in range(top + 1):
            assert node_id in index.graph[layer], \
                f"node {node_id} present at layer {top} but missing at layer {layer}"
    print("[PASS] every node present at its top layer is present at all layers below it")


def check_connectivity_from_entry_point(index, n):
    reachable = {index.entry_point}
    frontier = [index.entry_point]
    layer0 = index.graph[0]
    while frontier:
        node = frontier.pop()
        for neighbor in layer0.get(node, ()):
            if neighbor not in reachable:
                reachable.add(neighbor)
                frontier.append(neighbor)

    unreachable = set(range(n)) - reachable
    assert not unreachable, \
        f"{len(unreachable)} nodes unreachable from entry point via layer-0 graph: {sorted(unreachable)[:10]}..."
    print(f"[PASS] all {n} nodes reachable from entry point via layer-0 traversal (BFS)")


def report_level_distribution(index, n):
    counts = [len(layer) for layer in index.graph]
    print(f"[INFO] layer sizes (node count per layer): {counts}")
    print(f"[INFO] max_level={index.max_level}, expected ~log_M(n)={np.log(n)/np.log(index.M):.2f}")
    for i in range(1, len(counts)):
        if counts[i - 1] > 0:
            ratio = counts[i] / counts[i - 1]
            print(f"[INFO] layer {i-1}->{i} shrink ratio: {ratio:.3f} (target ~{1/index.M:.3f})")


def check_nearest_neighbor_sanity(index, vectors, n):
    query_id = 0
    query = vectors[query_id]
    dists = np.sum((vectors - query) ** 2, axis=1)
    true_nearest = set(np.argsort(dists)[:10].tolist())

    layer0_neighbors = set(index.graph[0][query_id])
    overlap = len(true_nearest & layer0_neighbors) if layer0_neighbors else 0
    print(f"[INFO] node 0's layer-0 graph neighbors overlap with its true 10-NN: "
          f"{overlap}/{min(10, len(layer0_neighbors))} "
          f"(informal sanity check, not a recall benchmark — that's Batch 2/5)")


def check_memory_bytes(index, n, dim):
    # Deterministic: calling it twice on the same, unmodified index must
    # give the exact same number -- this is the whole reason Batch 6
    # replaced ru_maxrss (which gave 0MB, 95MB, and 264MB across three
    # back-to-back runs of the identical build) with this method.
    first = index.memory_bytes()
    second = index.memory_bytes()
    assert first == second, \
        f"memory_bytes() is not deterministic: {first} != {second}"

    # Lower bound: the raw vector data alone (n vectors x dim float32s)
    # must already be present, since self.vectors holds owned copies.
    raw_vector_bytes = n * dim * 4
    assert first > raw_vector_bytes, \
        f"memory_bytes()={first} is below the raw vector data floor ({raw_vector_bytes})"

    # Monotonic: an index with more nodes must report more memory than one
    # with fewer, using the same dim/params -- catches a memory_bytes()
    # that accidentally measures something size-independent (e.g. a fixed
    # per-object overhead only).
    smaller_index, _ = build_index(n // 2, dim)
    assert smaller_index.memory_bytes() < first, \
        "a smaller index must report less memory than a larger one"

    print(f"[PASS] memory_bytes() deterministic ({first:,} bytes = "
          f"{first / (1024 * 1024):.1f}MB), above raw-vector floor, "
          f"and grows with index size")


if __name__ == "__main__":
    N, DIM = 2000, 50
    print(f"Building HNSW index: n={N}, dim={DIM}\n")
    index, vectors = build_index(N, DIM)

    check_layer0_completeness(index, N)
    check_bidirectional_edges(index)
    check_degree_limits(index)
    check_layer_membership_consistency(index, N)
    check_connectivity_from_entry_point(index, N)
    report_level_distribution(index, N)
    check_nearest_neighbor_sanity(index, vectors, N)
    check_memory_bytes(index, N, DIM)

    print("\nAll Batch 1 structural checks passed.")
