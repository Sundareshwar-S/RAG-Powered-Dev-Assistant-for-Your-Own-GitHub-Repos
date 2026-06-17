---
name: file-structure
description: File structure specialist for the project. Use PROACTIVELY when creating new files or directories.
tools: ["Read", "Write", "Edit", "Bash", "Grep", "Glob"]
---



├── docker-compose.yml
├── .env
├── .env.example
├── .gitignore
├── README.md
├── scripts/
│   └── start-ollama.sh    (starts Ollama with OLLAMA_MODELS=./ollama_models/models)
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes_ingest.py
│   │   ├── routes_query.py
│   │   └── routes_repos.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── git_cloner.py
│   │   ├── file_walker.py
│   │   ├── ast_chunker.py
│   │   ├── embedding_service.py
│   │   ├── chroma_writer.py
│   │   └── bm25_builder.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid_retriever.py
│   │   ├── dense_retriever.py
│   │   ├── sparse_retriever.py
│   │   ├── rrf_fusion.py
│   │   ├── reranker.py
│   │   ├── prompt_builder.py
│   │   └── ollama_client.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── dependencies.py
│   │   └── logger.py
│   └── jobs/
│       ├── __init__.py
│       └── job_store.py
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   ├── public/
│   │   └── favicon.ico
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── components/
│       │   ├── Sidebar/
│       │   │   ├── RepoManager.jsx
│       │   │   ├── IngestionProgress.jsx
│       │   │   └── IndexedRepoList.jsx
│       │   ├── Chat/
│       │   │   ├── ChatWindow.jsx
│       │   │   ├── UserMessage.jsx
│       │   │   ├── AssistantMessage.jsx
│       │   │   ├── StreamingCursor.jsx
│       │   │   └── QueryInput.jsx
│       │   └── Sources/
│       │       ├── SourceCards.jsx
│       │       └── CodeSnippetCard.jsx
│       ├── hooks/
│       │   ├── useStreamingChat.js
│       │   ├── useIngestion.js
│       │   └── useRepos.js
│       ├── services/
│       │   └── api.js
│       └── styles/
│           └── index.css
├── tests/
│   ├── golden_qa/
│   │   └── README.md
│   ├── eval_retrieval.py
│   ├── eval_generation.py
│   ├── load_test.py
│   └── conftest.py
├── chroma_data/           (gitignored — Docker volume)
├── bm25_cache/            (gitignored — Docker volume, persists BM25 indexes across restarts)
├── ollama_models/         (gitignored — local + Docker model storage)
│   ├── models/            (blobs + manifests; set OLLAMA_MODELS to this path)
│   └── ollama.log         (gitignored — local serve log)
└── repos_cache/           (gitignored — ephemeral clones)