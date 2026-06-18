import { useState, useCallback } from 'react'
import RepoManager from './components/Sidebar/RepoManager.jsx'
import IndexedRepoList from './components/Sidebar/IndexedRepoList.jsx'
import ChatWindow from './components/Chat/ChatWindow.jsx'
import { useStreamingChat } from './hooks/useStreamingChat.js'
import { useRepos } from './hooks/useRepos.js'

export default function App() {
  const { repos, loading, error: reposError, refresh, removeRepo } = useRepos()
  const { messages, sendMessage, isStreaming, streamingPhase } = useStreamingChat()
  const [selectedRepoId, setSelectedRepoId] = useState(null)

  // After ingestion finishes, refresh the repo list and auto-select the new repo
  const handleIngestComplete = useCallback(() => {
    refresh()
  }, [refresh])

  const handleDeleteRepo = useCallback(async (repoId) => {
    await removeRepo(repoId)
    if (selectedRepoId === repoId) {
      setSelectedRepoId(null)
    }
  }, [removeRepo, selectedRepoId])

  return (
    <div className="app">
      {/* ---- Sidebar ---- */}
      <aside className="sidebar">
        <div className="sidebar__brand">
          <span className="sidebar__logo">◈</span>
          <span className="sidebar__name">CodeBase Oracle</span>
        </div>

        <div className="sidebar__section">
          <RepoManager onIngestComplete={handleIngestComplete} />
        </div>

        <div className="sidebar__section sidebar__section--grow">
          <IndexedRepoList
            repos={repos}
            loading={loading}
            error={reposError}
            onDelete={handleDeleteRepo}
            onSelect={setSelectedRepoId}
            selectedRepoId={selectedRepoId}
          />
        </div>
      </aside>

      {/* ---- Main chat area ---- */}
      <main className="main">
        <ChatWindow
          messages={messages}
          isStreaming={isStreaming}
          streamingPhase={streamingPhase}
          onSend={sendMessage}
          repos={repos}
          selectedRepoId={selectedRepoId}
          onSelectRepo={setSelectedRepoId}
        />
      </main>
    </div>
  )
}
