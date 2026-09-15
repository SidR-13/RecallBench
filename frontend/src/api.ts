import axios from 'axios'
import type { BenchmarkResponse, CompareResponse, QueryResponse } from './types'

// VITE_API_URL, not a hardcoded localhost -- same convention as
// BuildBoard's frontend build (see its .github/workflows/deploy.yml),
// so this points at localhost:8000 in dev and a real deployed backend
// URL in production without a code change.
const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const client = axios.create({ baseURL: BASE_URL })

export async function postQuery(query: string, k: number, ef: number): Promise<QueryResponse> {
  const res = await client.post<QueryResponse>('/api/query', { query, k, ef })
  return res.data
}

export async function postCompare(query: string, k: number, ef: number): Promise<CompareResponse> {
  const res = await client.post<CompareResponse>('/api/compare', { query, k, ef })
  return res.data
}

export async function getBenchmark(): Promise<BenchmarkResponse> {
  const res = await client.get<BenchmarkResponse>('/api/benchmark')
  return res.data
}
