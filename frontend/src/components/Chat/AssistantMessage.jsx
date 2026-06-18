import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import StreamingCursor from './StreamingCursor.jsx'
import SourceCards from '../Sources/SourceCards.jsx'

const markdownComponents = {
  code({ node, inline, className, children, ...props }) {
    const match = /language-(\w+)/.exec(className || '')
    if (!inline && match) {
      return (
        <SyntaxHighlighter
          style={oneDark}
          language={match[1]}
          PreTag="div"
          customStyle={{ borderRadius: '6px', margin: '0.5rem 0', fontSize: '0.85rem' }}
          {...props}
        >
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      )
    }
    return (
      <code className="inline-code" {...props}>
        {children}
      </code>
    )
  },
}

export default function AssistantMessage({ content, sources, streaming }) {
  return (
    <div className="message message--assistant">
      <div className="message__avatar">AI</div>
      <div className="message__body">
        <div className="message__bubble message__bubble--assistant">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={markdownComponents}
          >
            {content}
          </ReactMarkdown>
          {streaming && <StreamingCursor />}
        </div>

        {sources && sources.length > 0 && !streaming && (
          <SourceCards sources={sources} />
        )}
      </div>
    </div>
  )
}
