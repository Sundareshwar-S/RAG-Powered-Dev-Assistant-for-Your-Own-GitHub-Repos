export default function IngestionProgress({ status, progress, currentFile, error }) {
  const percent = Math.round((progress || 0) * 100)

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
            <span className="progress-label">{percent}%</span>
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
