export default function IngestionProgress({ status, progress, phase, currentFile, error }) {
  const percent = Math.round((progress || 0) * 100)

  const phaseLabel = {
    chunking: 'Chunking',
    embedding: 'Embedding',
    bm25: 'Indexing',
    completed: 'Complete',
  }[phase] || null

  return (
    <div className="ingestion-progress">
      <div className="progress-bar-track">
        <div
          className={`progress-bar-fill ${status === 'failed' ? 'progress-bar-fill--error' : ''}`}
          style={{ width: `${percent}%` }}
        />
      </div>

      <div className="progress-meta">
        {status === 'completed' && (
          <span className="progress-label progress-label--success">
            ✓ Indexing complete
          </span>
        )}
        {status === 'failed' && (
          <span className="progress-label progress-label--error">
            ✗ {error || 'Indexing failed'}
          </span>
        )}
        {status === 'running' && (
          <>
            <span className="progress-label">
              {phaseLabel ? `${phaseLabel} · ${percent}%` : `${percent}%`}
            </span>
            {currentFile && (
              <span className="progress-file" title={currentFile}>
                {currentFile}
              </span>
            )}
          </>
        )}
      </div>
    </div>
  )
}
