from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/benchmark")
def benchmark_endpoint(request: Request):
    """Serves the already-measured comparison_full100k.json (Batch 6) as
    is -- NOT a live re-run. Rerunning the full 100k-corpus benchmark
    (build + ef-sweep for both indices) takes several minutes; a benchmark
    endpoint that did that per request would make the "resume payload"
    numbers non-reproducible under normal API load and would time out any
    reasonable HTTP client anyway.
    """
    return request.app.state.benchmark_results
