/**
 * useStreamingChat — WebSocket-backed streaming chat hook.
 *
 * Usage:
 *   const { messages, sendMessage, isStreaming, streamingPhase } = useStreamingChat()
 *
 * Call sendMessage({ repoId, question, model }) to send a query.
 * Messages accumulate in the `messages` array:
 *   { role: 'user' | 'assistant', content: string, sources?: SourceChunk[] }
 */

import { useState, useRef, useCallback } from 'react'

const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000'

export function useStreamingChat() {
  const [messages, setMessages] = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingPhase, setStreamingPhase] = useState(null)
  const wsRef = useRef(null)
  const gotDoneRef = useRef(false)

  const _closeWs = () => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }

  const _finishStreaming = (updater) => {
    gotDoneRef.current = true
    setStreamingPhase(null)
    setIsStreaming(false)
    if (updater) {
      setMessages(updater)
    }
    _closeWs()
  }

  const sendMessage = useCallback(({ repoId, question, model }) => {
    if (isStreaming) return
    if (!repoId || !question.trim()) return

    setMessages((prev) => [...prev, { role: 'user', content: question }])
    setMessages((prev) => [...prev, { role: 'assistant', content: '', sources: [], streaming: true }])

    gotDoneRef.current = false
    setStreamingPhase('connecting')
    setIsStreaming(true)
    _closeWs()

    const ws = new WebSocket(`${WS_BASE_URL}/api/v1/ws/chat`)
    wsRef.current = ws

    ws.onopen = () => {
      setStreamingPhase('loading_index')
      ws.send(JSON.stringify({ repo_id: repoId, question, model }))
    }

    ws.onmessage = (event) => {
      let data
      try {
        data = JSON.parse(event.data)
      } catch {
        return
      }

      if (data.type === 'status') {
        setStreamingPhase(data.phase || 'retrieving')
      } else if (data.type === 'sources') {
        setMessages((prev) => {
          const updated = [...prev]
          const last = updated[updated.length - 1]
          if (last && last.role === 'assistant') {
            updated[updated.length - 1] = { ...last, sources: data.sources }
          }
          return updated
        })
      } else if (data.type === 'token') {
        setStreamingPhase('generating')
        setMessages((prev) => {
          const updated = [...prev]
          const last = updated[updated.length - 1]
          if (last && last.role === 'assistant') {
            updated[updated.length - 1] = {
              ...last,
              content: last.content + data.token,
            }
          }
          return updated
        })
      } else if (data.type === 'done') {
        _finishStreaming((prev) => {
          const updated = [...prev]
          const last = updated[updated.length - 1]
          if (last && last.role === 'assistant') {
            updated[updated.length - 1] = { ...last, streaming: false }
          }
          return updated
        })
      } else if (data.type === 'error') {
        _finishStreaming((prev) => {
          const updated = [...prev]
          const last = updated[updated.length - 1]
          if (last && last.role === 'assistant') {
            updated[updated.length - 1] = {
              ...last,
              content: `Error: ${data.message}`,
              streaming: false,
            }
          }
          return updated
        })
      }
    }

    ws.onerror = () => {
      if (gotDoneRef.current) return
      _finishStreaming((prev) => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last && last.role === 'assistant' && last.streaming) {
          updated[updated.length - 1] = {
            ...last,
            content: last.content || 'Connection error. Please try again.',
            streaming: false,
          }
        }
        return updated
      })
    }

    ws.onclose = () => {
      wsRef.current = null
      if (gotDoneRef.current) return
      _finishStreaming((prev) => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last && last.role === 'assistant' && last.streaming) {
          updated[updated.length - 1] = {
            ...last,
            content: last.content || (
              'The server stopped responding (often due to low memory). '
              + 'Try a shorter question or restart with `docker compose up --build`.'
            ),
            streaming: false,
          }
        }
        return updated
      })
    }
  }, [isStreaming])

  const clearMessages = useCallback(() => {
    _closeWs()
    setMessages([])
    setIsStreaming(false)
    setStreamingPhase(null)
    gotDoneRef.current = false
  }, [])

  return { messages, sendMessage, isStreaming, streamingPhase, clearMessages }
}
