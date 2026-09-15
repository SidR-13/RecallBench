from anthropic import Anthropic

from core.config import settings

# Instructs Claude to answer ONLY from the retrieved chunks, not its own
# general knowledge, and to say so explicitly when the context doesn't
# contain the answer. This keeps the demo honest about what it's actually
# testing: retrieval quality, not the model's training data. A prompt that
# let Claude freely mix in outside knowledge would make it impossible to
# tell, from the answer alone, whether retrieval actually worked.
_SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages "
    "provided below. If the passages do not contain enough information "
    "to answer the question, say so explicitly instead of guessing or "
    "using outside knowledge. When you do answer, mention which passage "
    "number(s) you drew on."
)


def retrieve(index, id_to_text, query_vec, k=5, ef=100):
    """Run the actual HNSW search and attach real corpus text to each
    result -- the retrieval half of RAG. Returns a list of dicts (not bare
    tuples) since build_prompt/generate_answer both need the text, not
    just the id/distance search() itself returns.
    """
    results = index.search(query_vec, k, ef=ef)
    return [
        {"id": node_id, "distance": dist, "text": id_to_text[node_id]}
        for node_id, dist in results
    ]


def build_prompt(query_text, retrieved):
    # Numbered passages (not just concatenated text) so the prompt can ask
    # Claude to cite which one(s) it used -- makes a wrong/ungrounded
    # answer visibly detectable instead of an unverifiable claim.
    passages = "\n\n".join(
        f"[{i + 1}] {chunk['text']}" for i, chunk in enumerate(retrieved)
    )
    return (
        f"Context passages:\n\n{passages}\n\n"
        f"Question: {query_text}"
    )


def _mock_answer(retrieved):
    # Same convention as BuildBoard's claude_service.py: a clearly labeled
    # [MOCK] response, so it can never be mistaken for a real model output,
    # and CI never needs a real ANTHROPIC_API_KEY or incurs API cost.
    passage_ids = ", ".join(str(chunk["id"]) for chunk in retrieved)
    return (
        f"[MOCK] AI_MOCK=true, no real Claude call made. "
        f"Retrieved passage ids: [{passage_ids}]. "
        f"Set AI_MOCK=false and provide a real ANTHROPIC_API_KEY for a real answer."
    )


_client = None


def _get_client():
    # Lazily constructed once per process and reused -- not one fresh
    # Anthropic() (and its underlying httpx connection pool) per request.
    # Found during an exhaustive review: the previous per-call
    # construction paid client/TLS-connection setup cost on every single
    # non-mock /api/query request instead of reusing a keep-alive
    # connection, the way any HTTP-based SDK is meant to be used.
    global _client
    if _client is None:
        _client = Anthropic(api_key=settings.anthropic_api_key)
    return _client


def generate_answer(query_text, retrieved):
    # Checked before the client is even constructed, so a placeholder/dev
    # key can't accidentally reach the network -- same ordering as
    # BuildBoard's analyze_failure().
    if settings.ai_mock:
        return _mock_answer(retrieved)

    client = _get_client()
    message = client.messages.create(
        model=settings.claude_model,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(query_text, retrieved)}],
    )
    return "".join(
        block.text for block in message.content if block.type == "text"
    )


def answer_query(index, id_to_text, embed_fn, query_text, k=5, ef=100):
    """Top-level RAG entry point: embed the query, retrieve real chunks
    from the real corpus via our own HNSW, and generate a grounded answer.

    `embed_fn` is injected (rather than importing core.embed directly)
    so tests can pass a cheap fake embedder and skip loading the real
    sentence-transformers model when they only care about prompt assembly
    or the AI_MOCK path -- the same "pass the function in" pattern
    benchmarks/harness.py already uses for insert_fn.
    """
    # show_progress=False: a tqdm bar for embedding a single short string
    # is pure overhead, and this call runs inside core/embed.py's
    # process-wide _model_lock -- every extra millisecond here is a
    # millisecond every other concurrent /api/query or /api/compare
    # request has to wait. compare.py already does this; this was the one
    # call site that didn't (see CLAUDE_CONTEXT.md's exhaustive-review log).
    query_vec = embed_fn([query_text], show_progress=False)[0]
    retrieved = retrieve(index, id_to_text, query_vec, k=k, ef=ef)
    answer = generate_answer(query_text, retrieved)
    return {"query": query_text, "retrieved": retrieved, "answer": answer}
