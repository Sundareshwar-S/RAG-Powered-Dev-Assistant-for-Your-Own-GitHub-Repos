import { useState } from 'react'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'

const LANGUAGE_ALIASES = {
  py: 'python',
  js: 'javascript',
  ts: 'typescript',
  jsx: 'jsx',
  tsx: 'tsx',
  rs: 'rust',
  go: 'go',
  java: 'java',
  md: 'markdown',
  txt: 'text',
}

function resolveLanguage(chunk) {
  if (chunk.language) return LANGUAGE_ALIASES[chunk.language] || chunk.language
  const ext = chunk.file_path?.split('.').pop()?.toLowerCase()
  return LANGUAGE_ALIASES[ext] || ext || 'text'
}

const CHUNK_TYPE_LABELS = {
  function_definition: 'fn',
  class_definition: 'class',
  function_declaration: 'fn',
  class_declaration: 'class',
  arrow_function: 'arrow',
  method_declaration: 'method',
  function_item: 'fn',
  impl_item: 'impl',
  snippet: 'snippet',
}

export default function CodeSnippetCard({ chunk }) {
  const [expanded, setExpanded] = useState(false)
  const language = resolveLanguage(chunk)
  const typeLabel = CHUNK_TYPE_LABELS[chunk.chunk_type] || chunk.chunk_type || 'snippet'

  const fileName = chunk.file_path?.split('/').pop() || 'unknown'
  const dirPath = chunk.file_path?.split('/').slice(0, -1).join('/') || ''

  return (
    <div className="snippet-card">
      <button
        className="snippet-card__header"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        aria-label={`${expanded ? 'Collapse' : 'Expand'} ${chunk.file_path}`}
      >
        <div className="snippet-card__meta">
          <span className="snippet-card__badge snippet-card__badge--type">{typeLabel}</span>
          <span className="snippet-card__badge snippet-card__badge--lang">{language}</span>
        </div>

        <div className="snippet-card__path">
          {dirPath && <span className="snippet-card__dir">{dirPath}/</span>}
          <span className="snippet-card__file">{fileName}</span>
        </div>

        <div className="snippet-card__lines">
          L{chunk.start_line}–{chunk.end_line}
        </div>

        <span className="snippet-card__toggle">{expanded ? '▲' : '▼'}</span>
      </button>

      {chunk.symbol_name && (
        <div className="snippet-card__symbol">{chunk.symbol_name}</div>
      )}

      {expanded && (
        <div className="snippet-card__code">
          <SyntaxHighlighter
            style={oneDark}
            language={language}
            showLineNumbers
            startingLineNumber={chunk.start_line || 1}
            customStyle={{
              margin: 0,
              borderRadius: '0 0 6px 6px',
              fontSize: '0.8rem',
              maxHeight: '300px',
            }}
          >
            {chunk.text || ''}
          </SyntaxHighlighter>
        </div>
      )}
    </div>
  )
}
