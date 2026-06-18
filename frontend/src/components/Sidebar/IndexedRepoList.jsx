export default function IndexedRepoList({ repos, loading, error, onDelete, onSelect, selectedRepoId }) {
  if (loading) {
    return <p className="list-placeholder">Loading repos…</p>
  }

  if (error) {
    return <p className="list-placeholder list-placeholder--error">{error}</p>
  }

  if (!repos || repos.length === 0) {
    return (
      <p className="list-placeholder">
        No repos indexed yet. Paste a GitHub URL above to get started.
      </p>
    )
  }

  return (
    <div className="repo-list">
      <h2 className="section-title">Indexed Repos</h2>
      <ul className="repo-list__items">
        {repos.map((repo) => (
          <RepoCard
            key={repo.repo_id}
            repo={repo}
            isSelected={repo.repo_id === selectedRepoId}
            onSelect={onSelect}
            onDelete={onDelete}
          />
        ))}
      </ul>
    </div>
  )
}

function RepoCard({ repo, isSelected, onSelect, onDelete }) {
  const shortUrl = repo.collection
    ? repo.collection.replace('repo_', '')
    : repo.repo_id

  return (
    <li
      className={`repo-card ${isSelected ? 'repo-card--selected' : ''}`}
      onClick={() => onSelect && onSelect(repo.repo_id)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && onSelect && onSelect(repo.repo_id)}
    >
      <div className="repo-card__body">
        <span className="repo-card__id" title={repo.repo_id}>
          {shortUrl}
        </span>
        <span className="repo-card__chunks">
          {repo.chunks >= 0 ? `${repo.chunks.toLocaleString()} chunks` : '—'}
          {repo.chunks >= 0 && repo.chunks <= 30 && (
            <span className="repo-card__hint" title="Low chunk count — re-index after updating ingest settings">
              {' '}· low
            </span>
          )}
        </span>
      </div>

      <button
        className="repo-card__delete"
        title="Delete repo"
        onClick={(e) => {
          e.stopPropagation()
          if (onDelete) onDelete(repo.repo_id)
        }}
        aria-label={`Delete repo ${repo.repo_id}`}
      >
        ✕
      </button>
    </li>
  )
}
