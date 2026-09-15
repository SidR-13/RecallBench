import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fastapi import APIRouter, Request

from app.schemas import RetrievalRequest
from core.embed import embed_texts
from core.rag import answer_query

router = APIRouter()


@router.post("/query")
def query_endpoint(body: RetrievalRequest, request: Request):
    """The real RAG pipeline: embed the query, retrieve real corpus text
    via our own hand-built HNSW (never FAISS -- see CLAUDE_CONTEXT.md,
    FAISS exists only as the benchmark comparison baseline), and generate
    a grounded Claude answer (or the labeled [MOCK] response when
    AI_MOCK=true).
    """
    return answer_query(
        request.app.state.hnsw_index,
        request.app.state.id_to_text,
        embed_texts,
        body.query,
        k=body.k,
        ef=body.ef,
    )
