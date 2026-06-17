import CodeSnippetCard from './CodeSnippetCard.jsx'

export default function SourceCards({ sources }) {
  if (!sources || sources.length === 0) return null

  return (
    <div className="source-cards">
      <p className="source-cards__label">
        {sources.length} source{sources.length !== 1 ? 's' : ''}
      </p>
      <div className="source-cards__scroll">
        {sources.map((chunk, idx) => (
          <CodeSnippetCard key={idx} chunk={chunk} />
        ))}
      </div>
    </div>
  )
}
