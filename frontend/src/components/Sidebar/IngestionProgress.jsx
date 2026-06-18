export default function IngestionProgress({
  status,
  progress,
  phase,
  currentFile,
  error,
  filesIndexed,
  chunksIndexed,
  filesSkipped,
}) {
  const percent = Math.round((progress || 0) * 100)

  const phaseLabel = {
    chunking: 'Chunking',
    embedding: 'Embedding',
    bm25: 'Indexing',
    completed: 'Complete',
  }[phase] || null

  const showLargeFileHint =
    status === 'running' &&
    phase === 'chunking' &&
    currentFile &&
    currentFile.startsWith('Chunking ')

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
          <>
            <span className="progress-label progress-label--success">
              ✓ Indexing complete
            </span>
            {filesIndexed != null && chunksIndexed != null && (
              <span className="progress-stats">
                {filesIndexed.toLocaleString()} files → {chunksIndexed.toLocaleString()} chunks
                {filesSkipped != null && filesSkipped > 0
                  ? ` (${filesSkipped.toLocaleString()} assets skipped)`
                  : ''}
              </span>
            )}
          </>
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
              <span className="progress-file progress-file--current" title={currentFile}>
                {currentFile}
              </span>
            )}
            {showLargeFileHint && (
              <span className="progress-hint">Large file in progress…</span>
            )}
          </>
        )}
      </div>
    </div>
  )
}
