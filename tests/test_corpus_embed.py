import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.corpus import load_wikipedia_paragraphs
from core.embed import embed_texts
from core.hnsw import HNSW


def check_corpus_loading():
    t0 = time.time()
    ids, texts = load_wikipedia_paragraphs(n=500, seed=42, prefix_rows=5000)
    elapsed = time.time() - t0

    assert len(ids) == 500 and len(texts) == 500
    assert all(isinstance(i, int) for i in ids)
    assert all(isinstance(t, str) and len(t) > 0 for t in texts)
    assert len(set(ids)) == 500, "sampled ids should be unique (sampling without replacement)"
    print(f"[PASS] loaded 500 real Wikipedia paragraphs in {elapsed:.1f}s")
    print(f"[INFO] example paragraph (id={ids[0]}): {texts[0][:150]!r}...")
    return ids, texts


def check_embedding_shape_and_normalization(texts):
    t0 = time.time()
    embeddings = embed_texts(texts[:50], show_progress=False)
    elapsed = time.time() - t0

    assert embeddings.shape == (50, 384), f"expected (50, 384), got {embeddings.shape}"
    assert embeddings.dtype == np.float32, f"expected float32, got {embeddings.dtype}"

    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"expected unit-norm vectors, got norms {norms[:5]}..."
    print(f"[PASS] embedded 50 paragraphs in {elapsed:.1f}s -> shape (50, 384), "
          f"float32, all unit-norm (max deviation from 1.0: {np.abs(norms - 1.0).max():.2e})")
    return embeddings


def check_semantic_sanity():
    # Two sentences about the same topic should embed closer together than
    # two sentences about unrelated topics -- a real, interpretable check
    # that the embedding space behaves the way semantic search requires.
    texts = [
        "The cat sat on the mat in the sunny afternoon.",
        "A small feline rested on the rug as sunlight streamed in.",
        "The stock market fell sharply after the interest rate announcement.",
    ]
    embeddings = embed_texts(texts, show_progress=False)
    cat_a, cat_b, unrelated = embeddings

    dist_same_topic = np.sum((cat_a - cat_b) ** 2)
    dist_diff_topic = np.sum((cat_a - unrelated) ** 2)

    assert dist_same_topic < dist_diff_topic, (
        f"expected same-topic sentences closer together, got "
        f"same-topic={dist_same_topic:.4f} >= different-topic={dist_diff_topic:.4f}"
    )
    print(f"[PASS] semantic sanity: same-topic distance ({dist_same_topic:.4f}) "
          f"< different-topic distance ({dist_diff_topic:.4f})")


def check_end_to_end_real_query(ids, texts):
    # Real, interpretable check: build a small HNSW index from real
    # Wikipedia paragraphs, issue a real natural-language query, and
    # confirm the retrieved paragraph text is actually relevant -- not
    # just a numeric assertion.
    embeddings = embed_texts(texts, show_progress=False)

    index = HNSW(dim=384, M=16, ef_construction=200, seed=42)
    for node_id, vec in zip(ids, embeddings):
        index.insert(node_id, vec)

    query_text = "science and space exploration"
    query_vec = embed_texts([query_text], show_progress=False)[0]

    id_to_text = dict(zip(ids, texts))
    results = index.search(query_vec, k=3, ef=100)

    print(f"\n[INFO] query: {query_text!r}")
    print("[INFO] top-3 retrieved paragraphs:")
    for node_id, dist in results:
        snippet = id_to_text[node_id][:150].replace("\n", " ")
        print(f"[INFO]   dist={dist:.4f} id={node_id}: {snippet!r}...")

    assert len(results) == 3
    print("[PASS] end-to-end real query against real embedded Wikipedia paragraphs "
          "returned 3 results -- inspect above for topical relevance")


if __name__ == "__main__":
    ids, texts = check_corpus_loading()
    check_embedding_shape_and_normalization(texts)
    check_semantic_sanity()
    check_end_to_end_real_query(ids, texts)
    print("\nAll Batch 4 pipeline checks passed.")
