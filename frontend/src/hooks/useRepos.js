/**
 * useRepos — repo list management hook.
 *
 * Fetches the list of indexed repositories on mount and provides
 * actions to refresh and delete repos.
 */

import { useState, useEffect, useCallback } from 'react'
import { getRepos, deleteRepo } from '../services/api.js'

export function useRepos() {
  const [repos, setRepos] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchRepos = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await getRepos()
      setRepos(data || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchRepos()
  }, [fetchRepos])

  const removeRepo = useCallback(async (repoId) => {
    try {
      await deleteRepo(repoId)
      setRepos((prev) => prev.filter((r) => r.repo_id !== repoId))
    } catch (err) {
      setError(err.message)
    }
  }, [])

  return {
    repos,
    loading,
    error,
    refresh: fetchRepos,
    removeRepo,
  }
}
