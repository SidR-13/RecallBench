import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fastapi import APIRouter, Request

from app.schemas import RetrievalRequest
from core.embed import embed_texts
from core.rag import retrieve

router = APIRouter()


@router.post("/compare")
def compare_endpoint(body: RetrievalRequest, request: Request):
    """Retrieves the SAME live query from both our HNSW and FAISS, side by
    side -- the "query box -> ... -> side-by-side comparison against
    FAISS on the same query" demo goal from CLAUDE_CONTEXT.md's "Why We
    Are Building This". No Claude call here: this endpoint is about
    comparing retrieval, not generation.

    Both sides go through the same retrieve() helper core/rag.py already
    defines -- it only requires an object exposing search(query, k, ef=),
    which both HNSW and FaissHNSW do. An earlier version hand-rolled the
    FAISS side's id/distance/text dict-building inline instead of reusing
    retrieve(); found during an exhaustive review that the two branches
    could silently drift out of shape from each other. Reusing one
    function for both means a future fix to retrieve() (e.g. the
    KeyError-safety fix in its docstring) automatically covers both index
    types, not just whichever one a caller remembered to update.
    """
    id_to_text = request.app.state.id_to_text
    query_vec = embed_texts([body.query], show_progress=False)[0]

    our_hnsw = retrieve(request.app.state.hnsw_index, id_to_text, query_vec, k=body.k, ef=body.ef)
    faiss_results = retrieve(request.app.state.faiss_index, id_to_text, query_vec, k=body.k, ef=body.ef)

    return {"query": body.query, "our_hnsw": our_hnsw, "faiss": faiss_results}
