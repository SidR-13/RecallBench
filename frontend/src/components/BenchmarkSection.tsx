import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { BenchmarkResponse } from '../types'

// Merges both indices' ef-sweeps into one array keyed by ef, so recharts
// can plot both lines on shared X-axis ticks without two separate charts.
//
// Keyed by the actual `ef` value, not array index -- an earlier version
// zipped `our_hnsw.ef_sweep[i]` with `faiss_hnsw.ef_sweep[i]` positionally,
// trusting the two independently-generated arrays to happen to share
// length and order. scripts/run_benchmark.py itself only trusts this
// with an explicit `assert ours["ef"] == theirs["ef"]` before zipping
// (see its printed comparison table) -- found during an exhaustive review
// that the frontend had no equivalent safety, so a benchmark JSON with
// mismatched-length or reordered sweeps would silently pair the wrong ef
// values with no error, exactly the kind of quiet wrongness this project
// exists to rule out.
function mergeSweeps(benchmark: BenchmarkResponse) {
  const faissByEf = new Map(benchmark.faiss_hnsw.ef_sweep.map((p) => [p.ef, p]))

  return benchmark.our_hnsw.ef_sweep.map((point) => {
    const faissPoint = faissByEf.get(point.ef)
    if (!faissPoint) {
      throw new Error(`benchmark results missing faiss_hnsw ef_sweep entry for ef=${point.ef}`)
    }
    return {
      ef: point.ef,
      our_recall: point.recall_at_k,
      faiss_recall: faissPoint.recall_at_k,
      our_p50: point.latency_p50_ms,
      faiss_p50: faissPoint.latency_p50_ms,
    }
  })
}

export default function BenchmarkSection({ benchmark }: { benchmark: BenchmarkResponse }) {
  const data = mergeSweeps(benchmark)

  return (
    <div className="rounded-lg border border-(--color-border) bg-(--color-surface) p-5">
      <h3 className="mb-1 text-sm font-medium text-(--color-text-muted)">
        Real benchmark — {benchmark.n.toLocaleString()} vectors, {benchmark.n_queries} held-out
        queries, k={benchmark.k}, M={benchmark.M}
      </h3>
      <p className="mb-4 text-xs text-(--color-text-muted)">
        Measured once (see CLAUDE_CONTEXT.md Batch 6), not re-run live per request.
      </p>

      <div className="mb-6 grid grid-cols-2 gap-4 text-sm sm:grid-cols-4">
        <Stat label="Build time (ours)" value={`${benchmark.our_hnsw.build_seconds.toFixed(1)}s`} color="var(--color-hnsw)" />
        <Stat label="Build time (FAISS)" value={`${benchmark.faiss_hnsw.build_seconds.toFixed(1)}s`} color="var(--color-faiss)" />
        <Stat label="Memory (ours)" value={`${benchmark.our_hnsw.index_memory_mb.toFixed(1)} MB`} color="var(--color-hnsw)" />
        <Stat label="Memory (FAISS)" value={`${benchmark.faiss_hnsw.index_memory_mb.toFixed(1)} MB`} color="var(--color-faiss)" />
      </div>

      <div className="mb-6 h-64 w-full">
        <ResponsiveContainer>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis dataKey="ef" stroke="var(--color-text-muted)" label={{ value: 'ef', position: 'insideBottom', offset: -2, fill: 'var(--color-text-muted)' }} />
            <YAxis yAxisId="recall" domain={[0, 1]} stroke="var(--color-text-muted)" />
            <Tooltip contentStyle={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)' }} />
            <Legend />
            <Line yAxisId="recall" type="monotone" dataKey="our_recall" name="our recall@k" stroke="var(--color-hnsw)" strokeWidth={2} />
            <Line yAxisId="recall" type="monotone" dataKey="faiss_recall" name="FAISS recall@k" stroke="var(--color-faiss)" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-(--color-border) text-left text-(--color-text-muted)">
            <th className="py-1">ef</th>
            <th>our recall@k</th>
            <th>our p50</th>
            <th>FAISS recall@k</th>
            <th>FAISS p50</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row) => (
            <tr key={row.ef} className="border-b border-(--color-border)">
              <td className="py-1">{row.ef}</td>
              <td>{row.our_recall.toFixed(4)}</td>
              <td>{row.our_p50.toFixed(3)}ms</td>
              <td>{row.faiss_recall.toFixed(4)}</td>
              <td>{row.faiss_p50.toFixed(3)}ms</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Stat({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div>
      <div className="text-xs text-(--color-text-muted)">{label}</div>
      <div className="text-lg font-medium" style={{ color }}>
        {value}
      </div>
    </div>
  )
}
