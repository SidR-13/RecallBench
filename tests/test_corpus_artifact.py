import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.embed import embed_texts

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_PATH = DATA_DIR / "corpus_embeddings.npy"
METADATA_PATH = DATA_DIR / "corpus_metadata.jsonl"


def load_artifact():
    embeddings = np.load(EMBEDDINGS_PATH)
    ids, texts = [], []
    with open(METADATA_PATH) as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["id"])
            texts.append(row["text"])
    return embeddings, ids, texts


def check_shapes_and_counts(embeddings, ids, texts):
    assert embeddings.shape == (100_000, 384), f"expected (100000, 384), got {embeddings.shape}"
    assert embeddings.dtype == np.float32
    assert len(ids) == 100_000 and len(texts) == 100_000
    print(f"[PASS] artifact shape (100000, 384) float32, 100000 metadata rows, all counts match")


def check_ids_unique():
    _, ids, _ = load_artifact()
    assert len(set(ids)) == len(ids), "duplicate ids found in corpus metadata"
    print(f"[PASS] all {len(ids)} corpus ids are unique")


def check_unit_norm(embeddings):
    norms = np.linalg.norm(embeddings, axis=1)
    max_dev = np.abs(norms - 1.0).max()
    assert max_dev < 1e-4, f"expected unit-norm embeddings, max deviation {max_dev}"
    print(f"[PASS] all 100000 stored embeddings are unit-norm (max deviation from 1.0: {max_dev:.2e})")


def check_row_order_consistency(embeddings, ids, texts):
    # The real risk with two separately-written files (embeddings.npy and
    # metadata.jsonl) is a silent order mismatch -- row i's embedding must
    # actually correspond to row i's text. Re-embed a random sample of the
    # *stored* texts fresh and confirm they match the *stored* embedding at
    # that same row -- this is the check that would catch e.g. a shuffle
    # or reordering bug introduced between embedding and saving.
    rng = np.random.default_rng(0)
    sample_rows = rng.choice(len(ids), size=20, replace=False)
    sample_texts = [texts[i] for i in sample_rows]

    fresh_embeddings = embed_texts(sample_texts, show_progress=False)
    stored_embeddings = embeddings[sample_rows]

    max_diff = np.abs(fresh_embeddings - stored_embeddings).max()
    assert max_diff < 1e-4, (
        f"re-embedding stored text doesn't match the stored embedding at the same row "
        f"(max diff {max_diff}) -- possible row-order mismatch between "
        f"corpus_embeddings.npy and corpus_metadata.jsonl"
    )
    print(f"[PASS] spot-checked 20 random rows: re-embedding stored text matches the "
          f"stored embedding at the same row (max diff: {max_diff:.2e}) -- "
          f"confirms embeddings.npy and metadata.jsonl are in the same order")


def check_content_is_real_text(texts):
    rng = np.random.default_rng(1)
    sample = [texts[i] for i in rng.choice(len(texts), size=5, replace=False)]
    for t in sample:
        assert len(t) > 20, f"suspiciously short paragraph: {t!r}"
    print(f"[PASS] spot-checked 5 random paragraphs are real, non-trivial text, e.g.:")
    print(f"       {sample[0][:120]!r}...")


if __name__ == "__main__":
    embeddings, ids, texts = load_artifact()
    check_shapes_and_counts(embeddings, ids, texts)
    check_ids_unique()
    check_unit_norm(embeddings)
    check_row_order_consistency(embeddings, ids, texts)
    check_content_is_real_text(texts)
    print("\nAll Batch 4 corpus-artifact integrity checks passed.")
