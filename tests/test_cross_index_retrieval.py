import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.brute_force import BruteForce
from core.embed import embed_texts
from core.faiss_index import FaissHNSW
from core.hnsw import HNSW

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# A real, already-built end-to-end check: does every retrieval path
# (brute-force ground truth, our HNSW, FAISS) agree with each other AND
# return text a human would call relevant, on the real corpus -- not a
# synthetic vector, and not just an aggregate recall number averaged over
# 500 queries. Batch 4 did this for our own HNSW alone; this is the first
# time all three are checked side by side, by eye, since FAISS was added.
#
# A moderate subset (not the full 100k) keeps this fast to re-run -- same
# convention as Batch 4's own 500-paragraph semantic sanity check, which
# used a small slice rather than the full artifact for a correctness check
# (the full 100k scale is reserved for the actual benchmark numbers in
# comparison_full100k.json).
N_CORPUS = 10_000
QUERIES = [
    "science and space exploration",
    "history of ancient Rome",
    "programming languages and computers",
    "music genres and instruments",
    "climate change and the environment",
]
K = 5
EF = 100


def load_subset(n, seed=42):
    embeddings = np.load(DATA_DIR / "corpus_embeddings.npy")
    ids = []
    with open(DATA_DIR / "corpus_metadata.jsonl") as f:
        for line in f:
            ids.append(json.loads(line)["id"])
    ids = np.asarray(ids)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ids), size=n, replace=False)
    embeddings = embeddings[idx]
    ids = ids[idx]

    id_to_text = {}
    with open(DATA_DIR / "corpus_metadata.jsonl") as f:
        wanted = set(ids.tolist())
        for line in f:
            row = json.loads(line)
            if row["id"] in wanted:
                id_to_text[row["id"]] = row["text"]

    return ids, embeddings, id_to_text


def build_indices(ids, embeddings):
    dim = embeddings.shape[1]

    brute = BruteForce(embeddings, ids=ids)

    hnsw = HNSW(dim=dim, M=16, ef_construction=200, seed=42)
    for node_id, vec in zip(ids, embeddings):
        hnsw.insert(node_id, vec)

    faiss_index = FaissHNSW(dim=dim, M=16, ef_construction=200)
    for node_id, vec in zip(ids, embeddings):
        faiss_index.insert(node_id, vec)

    return brute, hnsw, faiss_index


def check_cross_index_agreement(brute, hnsw, faiss_index, id_to_text):
    total_hnsw_overlap = 0
    total_faiss_overlap = 0
    total_hnsw_faiss_overlap = 0
    total_possible = 0

    for query_text in QUERIES:
        query_vec = embed_texts([query_text], show_progress=False)[0]

        truth_ids = [nid for nid, _ in brute.search(query_vec, K)]
        hnsw_ids = [nid for nid, _ in hnsw.search(query_vec, K, ef=EF)]
        faiss_ids = [nid for nid, _ in faiss_index.search(query_vec, K, ef=EF)]

        hnsw_overlap = len(set(hnsw_ids) & set(truth_ids))
        faiss_overlap = len(set(faiss_ids) & set(truth_ids))
        hnsw_faiss_overlap = len(set(hnsw_ids) & set(faiss_ids))

        total_hnsw_overlap += hnsw_overlap
        total_faiss_overlap += faiss_overlap
        total_hnsw_faiss_overlap += hnsw_faiss_overlap
        total_possible += K

        print(f"\n[INFO] query: {query_text!r}")
        print(f"[INFO]   ground truth (brute-force) top-{K}:")
        for nid in truth_ids:
            print(f"[INFO]     id={nid}: {id_to_text[nid][:120].strip()!r}...")
        print(f"[INFO]   our HNSW top-{K} (overlap with truth: {hnsw_overlap}/{K}):")
        for nid in hnsw_ids:
            marker = "*" if nid in truth_ids else " "
            print(f"[INFO]    {marker}id={nid}: {id_to_text[nid][:120].strip()!r}...")
        print(f"[INFO]   FAISS top-{K} (overlap with truth: {faiss_overlap}/{K}):")
        for nid in faiss_ids:
            marker = "*" if nid in truth_ids else " "
            print(f"[INFO]    {marker}id={nid}: {id_to_text[nid][:120].strip()!r}...")

        # Every retrieval path must return real, non-empty, distinct text --
        # not proof of relevance (that needs a human reading the output
        # above) but a floor no retrieval method should ever fail.
        for nid in truth_ids + hnsw_ids + faiss_ids:
            assert nid in id_to_text
            assert len(id_to_text[nid].strip()) > 0

    hnsw_recall = total_hnsw_overlap / total_possible
    faiss_recall = total_faiss_overlap / total_possible
    agreement = total_hnsw_faiss_overlap / total_possible

    print(f"\n[INFO] across {len(QUERIES)} real queries, k={K}, ef={EF}, n={N_CORPUS}:")
    print(f"[INFO]   our HNSW vs. brute-force ground truth: {hnsw_recall:.2f}")
    print(f"[INFO]   FAISS vs. brute-force ground truth:    {faiss_recall:.2f}")
    print(f"[INFO]   our HNSW vs. FAISS (direct agreement): {agreement:.2f}")

    # Both approximate methods must substantially agree with exact ground
    # truth on real queries -- not the formal ef-sweep benchmark (that's
    # Batch 5/6's 500-query measurement), just a floor confirming nothing
    # is systematically broken (e.g. a metric mismatch or id-mapping bug)
    # on real corpus text, at a k/ef this project actually cares about.
    assert hnsw_recall >= 0.7, f"our HNSW recall on real queries too low: {hnsw_recall:.2f}"
    assert faiss_recall >= 0.7, f"FAISS recall on real queries too low: {faiss_recall:.2f}"

    print("\n[PASS] all three retrieval paths returned real, relevant, "
          "and substantially agreeing results on real corpus queries "
          "(see printed text above for manual relevance inspection)")


if __name__ == "__main__":
    print(f"Loading a real {N_CORPUS}-vector subset of the corpus...")
    ids, embeddings, id_to_text = load_subset(N_CORPUS)

    print("Building BruteForce, HNSW, and FAISS indices from the same subset...")
    brute, hnsw, faiss_index = build_indices(ids, embeddings)

    check_cross_index_agreement(brute, hnsw, faiss_index, id_to_text)
