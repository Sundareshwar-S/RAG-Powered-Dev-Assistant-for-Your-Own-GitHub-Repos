import { useEffect, useRef } from 'react'
import UserMessage from './UserMessage.jsx'
import AssistantMessage from './AssistantMessage.jsx'
import QueryInput from './QueryInput.jsx'

export default function ChatWindow({ messages, isStreaming, streamingPhase, onSend, repos, selectedRepoId, onSelectRepo }) {
  const endRef = useRef(null)

  const phaseLabel = {
    connecting: 'Connecting…',
    loading_index: 'Loading index…',
    retrieving: 'Searching codebase…',
    embedding: 'Embedding query…',
    reranking: 'Ranking results…',
    ranking: 'Ranking results…',
    generating: 'Generating…',
  }[streamingPhase] || 'Thinking…'

  // Auto-scroll to bottom whenever messages update
  useEffect(() => {
    if (endRef.current) {
      endRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' })
    }
  }, [messages])

  return (
    <div className="chat-window">
      <div className="chat-window__header">
        <span className="chat-window__title">CodeBase Oracle</span>
        {isStreaming && <span className="chat-window__streaming-badge">{phaseLabel}</span>}
      </div>

      <div className="chat-window__messages">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <p className="chat-empty__title">Ready to answer questions about your codebase</p>
            <p className="chat-empty__hint">Index a repo in the sidebar, select it below, then ask away.</p>
          </div>
        ) : (
          messages.map((msg, idx) => {
            if (msg.role === 'user') {
              return <UserMessage key={idx} content={msg.content} />
            }
            return (
              <AssistantMessage
                key={idx}
                content={msg.content}
                sources={msg.sources}
                streaming={msg.streaming}
              />
            )
          })
        )}
        <div ref={endRef} />
      </div>

      <div className="chat-window__input">
        <QueryInput
          repos={repos}
          selectedRepoId={selectedRepoId}
          onSelectRepo={onSelectRepo}
          onSend={onSend}
          disabled={isStreaming}
        />
      </div>
    </div>
  )
}
