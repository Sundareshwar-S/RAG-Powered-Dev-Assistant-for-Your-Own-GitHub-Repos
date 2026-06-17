/**
 * useIngestion — SSE-backed ingestion progress hook.
 *
 * Usage:
 *   const { startIngestion, progress, currentFile, status, error, isIngesting } = useIngestion()
 *
 * Call startIngestion(repoUrl, branch) to kick off an ingest.
 * The hook connects to the SSE stream and updates state until the job finishes.
 */

import { useState, useRef, useCallback, useEffect } from 'react'
import { ingestRepo } from '../services/api.js'

const BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

export function useIngestion(onComplete) {
  const [status, setStatus] = useState('idle')   // 'idle' | 'running' | 'completed' | 'failed'
  const [progress, setProgress] = useState(0)
  const [currentFile, setCurrentFile] = useState('')
  const [error, setError] = useState(null)
  const [jobId, setJobId] = useState(null)

  const esRef = useRef(null)

  // Clean up EventSource on unmount
  useEffect(() => {
    return () => {
      if (esRef.current) {
        esRef.current.close()
        esRef.current = null
      }
    }
  }, [])

  const _connectSSE = useCallback((id) => {
    if (esRef.current) {
      esRef.current.close()
    }

    const url = `${BASE_URL}/api/v1/ingest/${id}/status`
    const es = new EventSource(url)
    esRef.current = es

    es.onmessage = (event) => {
      let data
      try {
        data = JSON.parse(event.data)
      } catch {
        return
      }

      setProgress(data.progress ?? 0)
      setCurrentFile(data.current_file ?? '')
      setStatus(data.status ?? 'running')

      if (data.status === 'completed') {
        setProgress(1)
        es.close()
        esRef.current = null
        if (onComplete) onComplete()
      } else if (data.status === 'failed') {
        setError(data.error || 'Ingestion failed')
        es.close()
        esRef.current = null
      }
    }

    es.onerror = () => {
      setStatus('failed')
      setError('Connection to progress stream lost')
      es.close()
      esRef.current = null
    }
  }, [onComplete])

  const startIngestion = useCallback(async (repoUrl, branch = 'main') => {
    setStatus('running')
    setProgress(0)
    setCurrentFile('')
    setError(null)
    setJobId(null)

    try {
      const result = await ingestRepo(repoUrl, branch)
      setJobId(result.job_id)
      _connectSSE(result.job_id)
    } catch (err) {
      setStatus('failed')
      setError(err.message)
    }
  }, [_connectSSE])

  const reset = useCallback(() => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
    }
    setStatus('idle')
    setProgress(0)
    setCurrentFile('')
    setError(null)
    setJobId(null)
  }, [])

  return {
    startIngestion,
    reset,
    status,
    progress,
    currentFile,
    error,
    jobId,
    isIngesting: status === 'running',
  }
}
