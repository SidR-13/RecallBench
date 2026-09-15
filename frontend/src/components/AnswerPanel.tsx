import type { QueryResponse } from '../types'

export default function AnswerPanel({ result }: { result: QueryResponse }) {
  const isMock = result.answer.startsWith('[MOCK]')

  return (
    <div className="flex flex-col gap-4 rounded-lg border border-(--color-border) bg-(--color-surface) p-5">
      <div>
        <h3 className="mb-1 text-sm font-medium text-(--color-text-muted)">
          Generated answer{isMock && ' (AI_MOCK=true — set a real ANTHROPIC_API_KEY for a live answer)'}
        </h3>
        <p className={isMock ? 'font-mono text-sm text-(--color-text-muted)' : 'text-(--color-text)'}>
          {result.answer}
        </p>
      </div>

      <div>
        <h3 className="mb-2 text-sm font-medium text-(--color-text-muted)">
          Retrieved passages (our HNSW)
        </h3>
        <ol className="flex flex-col gap-2">
          {result.retrieved.map((chunk, i) => (
            <li
              key={chunk.id}
              className="rounded border border-(--color-border) px-3 py-2 text-sm"
            >
              <span className="mr-2 font-mono text-(--color-hnsw)">[{i + 1}]</span>
              {chunk.text}
              <span className="ml-2 font-mono text-xs text-(--color-text-muted)">
                dist={chunk.distance.toFixed(3)}
              </span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  )
}
