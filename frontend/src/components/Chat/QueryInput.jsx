import { useState, useRef } from 'react'

const MODELS = [
  { value: 'qwen2.5-coder:7b', label: 'Qwen2.5-Coder 7B' },
]

export default function QueryInput({ repos, selectedRepoId, onSelectRepo, onSend, disabled }) {
  const [question, setQuestion] = useState('')
  const [model, setModel] = useState(MODELS[0].value)
  const textareaRef = useRef(null)

  function handleSubmit(e) {
    e.preventDefault()
    const q = question.trim()
    if (!q || !selectedRepoId || disabled) return
    onSend({ repoId: selectedRepoId, question: q, model })
    setQuestion('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  function handleInput(e) {
    setQuestion(e.target.value)
    // Auto-resize textarea
    const ta = e.target
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`
  }

  const canSend = question.trim().length > 0 && !!selectedRepoId && !disabled

  return (
    <form className="query-input" onSubmit={handleSubmit}>
      <div className="query-input__controls">
        <select
          className="query-select"
          value={selectedRepoId || ''}
          onChange={(e) => onSelectRepo && onSelectRepo(e.target.value)}
          disabled={disabled}
          aria-label="Select repository"
        >
          <option value="" disabled>Select repo…</option>
          {(repos || []).map((r) => (
            <option key={r.repo_id} value={r.repo_id}>
              {r.repo_id}
            </option>
          ))}
        </select>

        <select
          className="query-select"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          disabled={disabled}
          aria-label="Select model"
        >
          {MODELS.map((m) => (
            <option key={m.value} value={m.value}>{m.label}</option>
          ))}
        </select>
      </div>

      <div className="query-input__row">
        <textarea
          ref={textareaRef}
          className="query-textarea"
          placeholder={selectedRepoId ? 'Ask anything about the codebase… (Enter to send)' : 'Select a repo first'}
          value={question}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          disabled={disabled || !selectedRepoId}
          rows={1}
          aria-label="Question"
        />
        <button
          type="submit"
          className="btn btn--send"
          disabled={!canSend}
          aria-label="Send"
        >
          {disabled ? '…' : '↑'}
        </button>
      </div>
    </form>
  )
}
