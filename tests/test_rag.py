import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import settings
from core.embed import embed_texts
from core.hnsw import HNSW
from core.rag import answer_query, build_prompt, generate_answer, retrieve

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
N_CORPUS = 3000  # small real subset -- this is a logic/correctness test,
# not the benchmark; full-scale HNSW builds are reserved for
# comparison_full100k.json.


def load_subset(n, seed=42):
    embeddings = np.load(DATA_DIR / "corpus_embeddings.npy")
    ids = []
    with open(DATA_DIR / "corpus_metadata.jsonl") as f:
        for line in f:
            ids.append(json.loads(line)["id"])
    ids = np.asarray(ids)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ids), size=n, replace=False)
    embeddings, ids = embeddings[idx], ids[idx]

    wanted = set(ids.tolist())
    id_to_text = {}
    with open(DATA_DIR / "corpus_metadata.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] in wanted:
                id_to_text[row["id"]] = row["text"]

    index = HNSW(dim=embeddings.shape[1], M=16, ef_construction=200, seed=42)
    for node_id, vec in zip(ids, embeddings):
        index.insert(node_id, vec)

    return index, id_to_text


def check_build_prompt_hand_checkable():
    # No real embedding/retrieval needed for this one -- prompt assembly
    # is pure string formatting, checkable with a fixed, fake input.
    retrieved = [
        {"id": 1, "distance": 0.1, "text": "Paris is the capital of France."},
        {"id": 2, "distance": 0.3, "text": "The Eiffel Tower is in Paris."},
    ]
    prompt = build_prompt("What is the capital of France?", retrieved)
    assert "[1] Paris is the capital of France." in prompt
    assert "[2] The Eiffel Tower is in Paris." in prompt
    assert "What is the capital of France?" in prompt
    print("[PASS] build_prompt includes numbered passages and the question")


def check_mock_answer_is_labeled_and_uses_real_ids():
    # Forces ai_mock True for the duration of this check regardless of the
    # ambient .env -- this test verifies the MOCK path specifically, not
    # whatever AI_MOCK happens to be set to in this environment. Now that
    # a real .env can set AI_MOCK=false (once a real ANTHROPIC_API_KEY is
    # added), this test must not assume the environment's default.
    original = settings.ai_mock
    settings.ai_mock = True
    try:
        retrieved = [{"id": 42, "distance": 0.0, "text": "irrelevant for this check"}]
        answer = generate_answer("does this matter?", retrieved)
        assert answer.startswith("[MOCK]")
        assert "42" in answer
    finally:
        settings.ai_mock = original
    print(f"[PASS] AI_MOCK path returns a labeled mock answer referencing real "
          f"retrieved ids: {answer!r}")


def check_retrieve_and_answer_query_end_to_end_real_corpus():
    # This check's purpose is retrieval correctness on the real corpus,
    # not the Claude call itself (check_live_claude_call_if_key_available
    # below is the one that exercises a real key when present) -- forces
    # ai_mock True for its duration so it stays deterministic and free
    # regardless of the ambient .env.
    index, id_to_text = load_subset(N_CORPUS)

    query_text = "What is the greenhouse effect?"
    query_vec = embed_texts([query_text], show_progress=False)[0]
    retrieved = retrieve(index, id_to_text, query_vec, k=3, ef=100)

    assert len(retrieved) == 3
    dists = [r["distance"] for r in retrieved]
    assert dists == sorted(dists), "retrieved chunks must be ascending by distance"
    for chunk in retrieved:
        assert chunk["id"] in id_to_text
        assert len(chunk["text"].strip()) > 0

    print(f"\n[INFO] query: {query_text!r}")
    print("[INFO] retrieved real passages:")
    for chunk in retrieved:
        print(f"[INFO]   dist={chunk['distance']:.4f} id={chunk['id']}: "
              f"{chunk['text'][:150].strip()!r}...")

    original = settings.ai_mock
    settings.ai_mock = True
    try:
        result = answer_query(index, id_to_text, embed_texts, query_text, k=3, ef=100)
    finally:
        settings.ai_mock = original
    assert result["query"] == query_text
    assert result["retrieved"] == retrieved
    assert result["answer"].startswith("[MOCK]")
    print(f"[INFO] answer (AI_MOCK path): {result['answer']!r}")
    print("[PASS] end-to-end retrieval on real corpus text succeeded; "
          "real Claude call path exercised via AI_MOCK here for determinism "
          "(the live path is checked separately below)")


def check_live_claude_call_if_key_available():
    if not settings.anthropic_api_key or settings.ai_mock:
        print("[INFO] SKIPPED: no real ANTHROPIC_API_KEY / AI_MOCK=false in "
              "this environment -- the live Claude call path is unverified. "
              "Not faking this result (see CLAUDE_CONTEXT.md rule 9).")
        return

    index, id_to_text = load_subset(N_CORPUS)
    result = answer_query(
        index, id_to_text, embed_texts, "What is the greenhouse effect?", k=3, ef=100
    )
    assert not result["answer"].startswith("[MOCK]")
    assert len(result["answer"].strip()) > 0
    print(f"[PASS] real live Claude call returned a real answer: {result['answer']!r}")


if __name__ == "__main__":
    check_build_prompt_hand_checkable()
    check_mock_answer_is_labeled_and_uses_real_ids()
    check_retrieve_and_answer_query_end_to_end_real_corpus()
    check_live_claude_call_if_key_available()
    print("\nAll Batch 7 RAG checks passed.")
