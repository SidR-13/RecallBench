import { useState } from 'react'

interface Props {
  onSubmit: (query: string, k: number, ef: number) => void
  loading: boolean
}

export default function QueryForm({ onSubmit, loading }: Props) {
  const [query, setQuery] = useState('')
  const [k, setK] = useState(5)
  const [ef, setEf] = useState(100)
  const [showAdvanced, setShowAdvanced] = useState(false)

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (query.trim().length === 0) return
    onSubmit(query.trim(), k, ef)
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3">
      <div className="flex gap-2">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask something answerable from the Wikipedia corpus..."
          className="flex-1 rounded-md border border-(--color-border) bg-(--color-surface) px-4 py-3 text-(--color-text) placeholder:text-(--color-text-muted) focus:outline-none focus:ring-1 focus:ring-(--color-hnsw)"
        />
        <button
          type="submit"
          disabled={loading || query.trim().length === 0}
          className="rounded-md bg-(--color-hnsw) px-5 py-3 font-medium text-black disabled:opacity-40"
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>

      <button
        type="button"
        onClick={() => setShowAdvanced((v) => !v)}
        className="self-start text-sm text-(--color-text-muted) underline"
      >
        {showAdvanced ? 'hide' : 'show'} advanced (k / ef)
      </button>

      {showAdvanced && (
        <div className="flex gap-4 text-sm text-(--color-text-muted)">
          <label className="flex items-center gap-2">
            k (results)
            <input
              type="number"
              min={1}
              max={20}
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="w-16 rounded border border-(--color-border) bg-(--color-surface) px-2 py-1 text-(--color-text)"
            />
          </label>
          <label className="flex items-center gap-2">
            ef (search width)
            <input
              type="number"
              min={1}
              max={1000}
              value={ef}
              onChange={(e) => setEf(Number(e.target.value))}
              className="w-20 rounded border border-(--color-border) bg-(--color-surface) px-2 py-1 text-(--color-text)"
            />
          </label>
        </div>
      )}
    </form>
  )
}
