import { useEffect, useState } from 'react'
import { getBenchmark, postCompare, postQuery } from './api'
import AnswerPanel from './components/AnswerPanel'
import BenchmarkSection from './components/BenchmarkSection'
import CompareView from './components/CompareView'
import QueryForm from './components/QueryForm'
import type { BenchmarkResponse, CompareResponse, QueryResponse } from './types'

function App() {
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null)
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null)
  const [benchmark, setBenchmark] = useState<BenchmarkResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Benchmark numbers are static (measured once, Batch 6) -- fetched once
  // on load, not re-fetched per query.
  useEffect(() => {
    getBenchmark()
      .then(setBenchmark)
      .catch(() => setError('Could not load benchmark results. Is the backend running?'))
  }, [])

  async function handleSearch(query: string, k: number, ef: number) {
    setLoading(true)
    setError(null)
    try {
      // Both real, independent calls to the same backend -- the answer
      // pipeline (our HNSW + Claude) and the retrieval-only comparison
      // (our HNSW vs FAISS) are separate endpoints (see
      // backend/app/routers/query.py vs compare.py), run in parallel here
      // since neither depends on the other's result.
      const [q, c] = await Promise.all([postQuery(query, k, ef), postCompare(query, k, ef)])
      setQueryResult(q)
      setCompareResult(c)
    } catch {
      setError('Search failed. Is the backend running at the configured VITE_API_URL?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-4xl flex-col gap-8 px-4 py-10">
      <header>
        <h1 className="text-3xl font-semibold text-(--color-text)">RecallBench</h1>
        <p className="mt-1 text-(--color-text-muted)">
          A hand-built HNSW vector index, retrieving from a real 100k-paragraph Wikipedia
          corpus, benchmarked honestly against FAISS.
        </p>
      </header>

      <QueryForm onSubmit={handleSearch} loading={loading} />

      {error && (
        <p className="rounded border border-red-800 bg-red-950/40 px-4 py-2 text-sm text-red-300">
          {error}
        </p>
      )}

      {queryResult && <AnswerPanel result={queryResult} />}
      {compareResult && <CompareView result={compareResult} />}

      <section>
        <h2 className="mb-3 text-xl font-medium text-(--color-text)">
          Offline benchmark: ours vs. FAISS
        </h2>
        {benchmark ? (
          <BenchmarkSection benchmark={benchmark} />
        ) : (
          !error && <p className="text-(--color-text-muted)">Loading benchmark results...</p>
        )}
      </section>
    </div>
  )
}

export default App
