/**
 * REST API wrappers for the CodeBase Oracle backend.
 *
 * All functions throw on non-2xx responses with an Error whose message
 * comes from the JSON body (field "error" or "detail"), falling back to
 * the HTTP status text.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })

  if (!response.ok) {
    let message = response.statusText
    try {
      const body = await response.json()
      message = body.error || body.detail || message
    } catch {
      // ignore JSON parse failure
    }
    const err = new Error(message)
    err.status = response.status
    throw err
  }

  // 204 No Content
  if (response.status === 204) return null

  return response.json()
}

/**
 * Start ingesting a repository.
 * Returns { job_id, status } or throws with err.status === 409 if already running.
 */
export async function ingestRepo(repoUrl, branch = 'main') {
  return request('/api/v1/ingest', {
    method: 'POST',
    body: JSON.stringify({ repo_url: repoUrl, branch }),
  })
}

/** List all indexed repositories. */
export async function getRepos() {
  return request('/api/v1/repos')
}

/** Delete an indexed repository by repo_id. */
export async function deleteRepo(repoId) {
  return request(`/api/v1/repos/${repoId}`, { method: 'DELETE' })
}

/**
 * Run a non-streaming RAG query.
 * Returns { answer, sources, model_used }.
 */
export async function queryRepo(repoId, question, model = null) {
  return request('/api/v1/query', {
    method: 'POST',
    body: JSON.stringify({ repo_id: repoId, question, model }),
  })
}

/** Check backend health. Returns { status, chroma, ollama }. */
export async function getHealth() {
  return request('/api/v1/health')
}
