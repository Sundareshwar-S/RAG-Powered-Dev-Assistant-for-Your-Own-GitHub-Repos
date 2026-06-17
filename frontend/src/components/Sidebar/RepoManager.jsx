import { useState } from 'react'
import { useIngestion } from '../../hooks/useIngestion.js'
import IngestionProgress from './IngestionProgress.jsx'

export default function RepoManager({ onIngestComplete }) {
  const [repoUrl, setRepoUrl] = useState('')
  const [branch, setBranch] = useState('main')
  const [conflict, setConflict] = useState(null)

  const { startIngestion, reset, status, progress, currentFile, error, isIngesting } =
    useIngestion(() => {
      if (onIngestComplete) onIngestComplete()
    })

  async function handleSubmit(e) {
    e.preventDefault()
    if (!repoUrl.trim()) return
    setConflict(null)

    try {
      await startIngestion(repoUrl.trim(), branch.trim() || 'main')
    } catch (err) {
      if (err && err.status === 409) {
        setConflict('This repo is already being indexed.')
      }
    }
  }

  function handleReset() {
    reset()
    setConflict(null)
  }

  const isDone = status === 'completed' || status === 'failed'

  return (
    <div className="repo-manager">
      <h2 className="section-title">Index Repository</h2>

      <form onSubmit={handleSubmit} className="repo-form">
        <div className="form-group">
          <label htmlFor="repo-url" className="form-label">GitHub URL</label>
          <input
            id="repo-url"
            type="url"
            className="form-input"
            placeholder="https://github.com/owner/repo"
            value={repoUrl}
            onChange={(e) => setRepoUrl(e.target.value)}
            disabled={isIngesting}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="branch" className="form-label">Branch</label>
          <input
            id="branch"
            type="text"
            className="form-input"
            placeholder="main"
            value={branch}
            onChange={(e) => setBranch(e.target.value)}
            disabled={isIngesting}
          />
        </div>

        {conflict && (
          <p className="alert alert--warning">{conflict}</p>
        )}

        <div className="form-actions">
          <button
            type="submit"
            className="btn btn--primary"
            disabled={isIngesting || !repoUrl.trim()}
          >
            {isIngesting ? 'Indexing…' : 'Index Repo'}
          </button>

          {isDone && (
            <button
              type="button"
              className="btn btn--ghost"
              onClick={handleReset}
            >
              Reset
            </button>
          )}
        </div>
      </form>

      {status !== 'idle' && (
        <IngestionProgress
          status={status}
          progress={progress}
          currentFile={currentFile}
          error={error}
        />
      )}
    </div>
  )
}
