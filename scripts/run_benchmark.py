"""Real benchmark run: build time, memory, and recall/latency across an ef
sweep, for both our hand-built HNSW and FAISS's IndexHNSWFlat, on the same
corpus split and the same held-out queries -- so every number in the final
comparison table comes from grading both indices against identical
ground truth, not two separately-run, potentially-inconsistent setups.

Both indices are built in this single process, one after the other. An
earlier version of this script built each index in its own fresh
subprocess, specifically to work around process-wide RSS (ru_maxrss)
memory measurement being contaminated by whatever else the process had
touched. That measurement approach was dropped entirely -- ru_maxrss
turned out to swing between 0MB, 95MB, and 264MB across three back-to-back
runs of the identical HNSW build (see CLAUDE_CONTEXT.md Batch 6 log).
Both HNSW.memory_bytes() and FaissHNSW.memory_bytes() now report exact,
deterministic byte counts of what each index actually owns, so there is no
longer any measurement reason to isolate the two builds into separate
processes.

Use --n for a smaller slice of the corpus, for a fast pilot run before
committing to the full corpus.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.harness import latency_percentiles, recall_at_k, time_build
from core.brute_force import BruteForce
from core.corpus import load_saved_corpus
from core.faiss_index import FaissHNSW
from core.hnsw import HNSW

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_PATH = DATA_DIR / "corpus_embeddings.npy"
METADATA_PATH = DATA_DIR / "corpus_metadata.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results"


def load_corpus(n, seed):
    ids, embeddings = load_saved_corpus(EMBEDDINGS_PATH, METADATA_PATH)

    if n < len(ids):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ids), size=n, replace=False)
        embeddings = embeddings[idx]
        ids = ids[idx]

    return ids, embeddings


def split_queries(ids, embeddings, n_queries, seed):
    rng = np.random.default_rng(seed + 1)
    query_pos = rng.choice(len(ids), size=n_queries, replace=False)
    query_mask = np.zeros(len(ids), dtype=bool)
    query_mask[query_pos] = True

    query_ids, query_vecs = ids[query_mask], embeddings[query_mask]
    index_ids, index_vecs = ids[~query_mask], embeddings[~query_mask]
    return index_ids, index_vecs, query_ids, query_vecs


def build_ground_truth(index_ids, index_vecs, query_vecs, k):
    """Computed once, before either index is built, since it's shared
    (exact, ef-independent) ground truth both HNSW and FAISS get graded
    against.
    """
    brute = BruteForce(index_vecs, ids=index_ids)
    return [[nid for nid, _ in brute.search(q, k)] for q in query_vecs]


def benchmark_one_index(index, index_ids, index_vecs, query_vecs, ground_truth,
                         ef_sweep, k):
    """Build `index`, time it, measure its memory, then sweep ef measuring
    recall@k and latency. Works identically for HNSW and FaissHNSW since
    both expose insert(node_id, vector) / search(query, k, ef) /
    memory_bytes().
    """
    build_seconds = time_build(index.insert, index_ids, index_vecs)
    index_mem_mb = index.memory_bytes() / (1024.0 * 1024.0)

    sweep_results = []
    for ef in ef_sweep:
        latencies = []
        recalls = []
        for q, truth in zip(query_vecs, ground_truth):
            t0 = time.perf_counter()
            results = index.search(q, k, ef=ef)
            latencies.append(time.perf_counter() - t0)
            recalls.append(recall_at_k([nid for nid, _ in results], truth))

        pct = latency_percentiles(latencies, percentiles=(50, 95))
        sweep_results.append({
            "ef": ef,
            "recall_at_k": float(np.mean(recalls)),
            "latency_p50_ms": pct[50],
            "latency_p95_ms": pct[95],
        })

    return {
        "build_seconds": build_seconds,
        "index_memory_mb": index_mem_mb,
        "ef_sweep": sweep_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100_000, help="corpus vectors to use")
    parser.add_argument("--n-queries", type=int, default=500, help="held-out query count")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-sweep", type=str, default="10,50,100,200,400")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="run", help="results filename tag")
    args = parser.parse_args()

    ef_sweep = [int(x) for x in args.ef_sweep.split(",")]

    print(f"Corpus: n={args.n}, n_queries={args.n_queries}, k={args.k}, "
          f"M={args.M}, ef_construction={args.ef_construction}, seed={args.seed}")

    ids, embeddings = load_corpus(args.n, args.seed)
    dim = embeddings.shape[1]
    index_ids, index_vecs, query_ids, query_vecs = split_queries(
        ids, embeddings, args.n_queries, args.seed
    )
    ground_truth = build_ground_truth(index_ids, index_vecs, query_vecs, args.k)

    hnsw = HNSW(dim=dim, M=args.M, ef_construction=args.ef_construction, seed=args.seed)
    hnsw_result = benchmark_one_index(
        hnsw, index_ids, index_vecs, query_vecs, ground_truth, ef_sweep, args.k
    )
    print(f"  [hnsw] build time: {hnsw_result['build_seconds']:.1f}s, "
          f"memory: {hnsw_result['index_memory_mb']:.1f} MB")

    faiss_index = FaissHNSW(dim=dim, M=args.M, ef_construction=args.ef_construction)
    faiss_result = benchmark_one_index(
        faiss_index, index_ids, index_vecs, query_vecs, ground_truth, ef_sweep, args.k
    )
    print(f"  [faiss] build time: {faiss_result['build_seconds']:.1f}s, "
          f"memory: {faiss_result['index_memory_mb']:.1f} MB")

    result = {
        "n": args.n,
        "n_queries": args.n_queries,
        "k": args.k,
        "M": args.M,
        "ef_construction": args.ef_construction,
        "seed": args.seed,
        "our_hnsw": hnsw_result,
        "faiss_hnsw": faiss_result,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"comparison_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {out_path}")

    print(f"\n{'ef':>6} | {'ours recall':>12} | {'ours p50':>9} | {'faiss recall':>13} | {'faiss p50':>10}")
    for ours, theirs in zip(hnsw_result["ef_sweep"], faiss_result["ef_sweep"]):
        assert ours["ef"] == theirs["ef"]
        print(f"{ours['ef']:>6} | {ours['recall_at_k']:>12.4f} | "
              f"{ours['latency_p50_ms']:>7.3f}ms | {theirs['recall_at_k']:>13.4f} | "
              f"{theirs['latency_p50_ms']:>8.3f}ms")
    print(f"\nbuild time -- ours: {hnsw_result['build_seconds']:.1f}s, "
          f"faiss: {faiss_result['build_seconds']:.1f}s")
    print(f"memory     -- ours: {hnsw_result['index_memory_mb']:.1f}MB, "
          f"faiss: {faiss_result['index_memory_mb']:.1f}MB")


if __name__ == "__main__":
    main()
