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

const BACKEND_LOST_MESSAGE =
  'Backend stopped responding (often out of memory during indexing). Rebuild with docker compose up --build and try again.'

async function fetchJobSnapshot(jobId) {
  const response = await fetch(`${BASE_URL}/api/v1/ingest/${jobId}`)
  if (!response.ok) {
    return null
  }
  return response.json()
}

export function useIngestion(onComplete) {
  const [status, setStatus] = useState('idle')   // 'idle' | 'running' | 'completed' | 'failed'
  const [progress, setProgress] = useState(0)
  const [phase, setPhase] = useState('')
  const [currentFile, setCurrentFile] = useState('')
  const [error, setError] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [filesIndexed, setFilesIndexed] = useState(null)
  const [chunksIndexed, setChunksIndexed] = useState(null)
  const [filesSkipped, setFilesSkipped] = useState(null)

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

  const _applyJobData = useCallback((data) => {
    setProgress(data.progress ?? 0)
    setPhase(data.phase ?? '')
    setCurrentFile(data.current_file ?? '')
    setStatus(data.status ?? 'running')
    if (data.files_indexed != null) setFilesIndexed(data.files_indexed)
    if (data.chunks_indexed != null) setChunksIndexed(data.chunks_indexed)
    if (data.files_skipped != null) setFilesSkipped(data.files_skipped)

    if (data.status === 'completed') {
      setProgress(1)
      if (onComplete) onComplete()
    } else if (data.status === 'failed') {
      setError(data.error || 'Ingestion failed')
    }
  }, [onComplete])

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

      _applyJobData(data)

      if (data.status === 'completed' || data.status === 'failed') {
        es.close()
        esRef.current = null
      }
    }

    es.onerror = async () => {
      es.close()
      esRef.current = null

      try {
        const snapshot = await fetchJobSnapshot(id)
        if (snapshot) {
          _applyJobData(snapshot)
          if (snapshot.status === 'completed' || snapshot.status === 'failed') {
            return
          }
        }
      } catch {
        // Backend unreachable — fall through to generic message
      }

      setStatus('failed')
      setError(BACKEND_LOST_MESSAGE)
    }
  }, [_applyJobData])

  const startIngestion = useCallback(async (repoUrl, branch = 'main') => {
    setStatus('running')
    setProgress(0)
    setPhase('')
    setCurrentFile('')
    setError(null)
    setJobId(null)
    setFilesIndexed(null)
    setChunksIndexed(null)
    setFilesSkipped(null)

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
    setPhase('')
    setCurrentFile('')
    setError(null)
    setJobId(null)
    setFilesIndexed(null)
    setChunksIndexed(null)
    setFilesSkipped(null)
  }, [])

  return {
    startIngestion,
    reset,
    status,
    progress,
    phase,
    currentFile,
    error,
    jobId,
    filesIndexed,
    chunksIndexed,
    filesSkipped,
    isIngesting: status === 'running',
  }
}
