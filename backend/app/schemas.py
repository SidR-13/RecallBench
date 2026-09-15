from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    """Shared by /api/query and /api/compare (query.py and compare.py) --
    both accept the identical query/k/ef shape, so one model instead of
    two independent copies.

    Bounds on k/ef are a real fix, not defensive paranoia: verified
    directly that k<=0 crashes FaissHNSW.search() (uncaught C++
    AssertionError) and silently misbehaves in HNSW.search() (negative
    Python slicing), and that core/hnsw.py's search() has no upper bound
    on ef -- an unbounded ef runs a full beam search over the entire
    100k-vector graph. Since query.py/compare.py are sync `def` routes,
    FastAPI runs them in its bounded thread pool; a few such requests can
    occupy every worker thread and stall the whole server for other users.
    Rejecting bad values here, at the actual network boundary, means
    core/hnsw.py's own floors (k<=0 -> [], ef floored at 1) are a second
    line of defense, not the only one.
    """

    query: str
    k: int = Field(5, gt=0, le=100)
    ef: int = Field(100, gt=0, le=2000)
