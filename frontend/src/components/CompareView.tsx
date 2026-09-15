import type { CompareResponse, RetrievedChunk } from '../types'

function ChunkList({
  title,
  color,
  chunks,
  otherIds,
}: {
  title: string
  color: string
  chunks: RetrievedChunk[]
  otherIds: Set<string | number>
}) {
  return (
    <div className="flex-1">
      <h3 className="mb-2 text-sm font-medium" style={{ color }}>
        {title}
      </h3>
      <ol className="flex flex-col gap-2">
        {chunks.map((chunk, i) => {
          const agrees = otherIds.has(chunk.id)
          return (
            <li
              key={chunk.id}
              className="rounded border px-3 py-2 text-sm"
              style={{ borderColor: agrees ? color : 'var(--color-border)' }}
            >
              <span className="mr-2 font-mono text-(--color-text-muted)">
                {i + 1}. {agrees ? '✓' : '·'}
              </span>
              {chunk.text}
            </li>
          )
        })}
      </ol>
    </div>
  )
}

export default function CompareView({ result }: { result: CompareResponse }) {
  const hnswIds = new Set(result.our_hnsw.map((c) => c.id))
  const faissIds = new Set(result.faiss.map((c) => c.id))
  const overlap = result.our_hnsw.filter((c) => faissIds.has(c.id)).length

  return (
    <div className="rounded-lg border border-(--color-border) bg-(--color-surface) p-5">
      <p className="mb-4 text-sm text-(--color-text-muted)">
        Same live query, retrieved independently from both indices — {overlap}/
        {result.our_hnsw.length} results agree (✓).
      </p>
      <div className="flex flex-col gap-4 md:flex-row">
        <ChunkList
          title="Our hand-built HNSW"
          color="var(--color-hnsw)"
          chunks={result.our_hnsw}
          otherIds={faissIds}
        />
        <ChunkList
          title="FAISS (IndexHNSWFlat)"
          color="var(--color-faiss)"
          chunks={result.faiss}
          otherIds={hnswIds}
        />
      </div>
    </div>
  )
}
