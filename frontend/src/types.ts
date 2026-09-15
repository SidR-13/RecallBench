// Mirrors the exact JSON shapes the backend (backend/app/routers/*.py)
// returns -- kept as one file so a backend response-shape change has one
// obvious place on the frontend to update too.

export interface RetrievedChunk {
  id: number | string
  distance: number
  text: string
}

export interface QueryResponse {
  query: string
  retrieved: RetrievedChunk[]
  answer: string
}

export interface CompareResponse {
  query: string
  our_hnsw: RetrievedChunk[]
  faiss: RetrievedChunk[]
}

export interface EfSweepPoint {
  ef: number
  recall_at_k: number
  latency_p50_ms: number
  latency_p95_ms: number
}

export interface IndexBenchmark {
  build_seconds: number
  index_memory_mb: number
  ef_sweep: EfSweepPoint[]
}

export interface BenchmarkResponse {
  n: number
  n_queries: number
  k: number
  M: number
  ef_construction: number
  seed: number
  our_hnsw: IndexBenchmark
  faiss_hnsw: IndexBenchmark
}
