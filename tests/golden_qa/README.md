# Golden QA Test Sets

This directory contains curated question-answer pairs for evaluating CodeBase Oracle's
retrieval and generation quality against specific GitHub repositories.

## Schema

Each JSON file is an array of QA entries:

```json
[
  {
    "question": "Natural-language question about the codebase",
    "expected_file": "path/to/expected/file.py",
    "expected_symbol": "function_or_class_name",
    "answer_keywords": ["keyword1", "keyword2", "keyword3"]
  }
]
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | yes | Natural-language question a developer would ask |
| `expected_file` | string | yes | Relative path within the repo that must appear in retrieval results |
| `expected_symbol` | string | yes | Function/class name that should appear in the retrieved chunk's `symbol_name` |
| `answer_keywords` | list[str] | yes | Substrings that must appear in the generated answer (case-insensitive) |

## Naming convention

Name each file `{repo_name}_qa.json` where `{repo_name}` is the repository's name in
lowercase (e.g. `markupsafe_qa.json` for `https://github.com/pallets/markupsafe`).

## How to write good questions

**Do:**
- Ask about specific, well-named functions or classes (`escape`, `Markup.unescape`)
- Include questions at different granularities: function-level, class-level, module-level
- Write questions as a developer would phrase them ("How does X work?", "What does Y return?")
- Include at least one question that spans two files (cross-file questions)
- Choose `answer_keywords` that must appear in any correct answer

**Don't:**
- Ask vague questions with no clear expected retrieval target ("What is this library?")
- Use keywords so rare they won't appear even in correct answers
- Duplicate the same question with minor wording changes

## Current test sets

| File | Repo | Questions | Distribution |
|------|------|-----------|--------------|
| `markupsafe_qa.json` | [pallets/markupsafe](https://github.com/pallets/markupsafe) | 25 | 10 function, 5 class, 5 module-level, 5 cross-file |

## Running evaluations

### Retrieval quality (Recall@5 and MRR)

```bash
# Requires a running stack (backend + Ollama + ChromaDB) with markupsafe indexed
python tests/eval_retrieval.py

# Offline integration tests (no running stack — uses temp ChromaDB + mocks)
pytest tests/test_phase1_integration.py tests/test_phase2_retrieval.py -v
```

Targets: **Recall@5 > 0.70**, **MRR > 0.60**

### Generation quality (keyword hit rate)

```bash
# Requires a running stack with markupsafe indexed
python tests/eval_generation.py
```

Target: **keyword_hit_rate > 0.80**

## Adding a new repository's test set

1. Index the repo: `POST /api/v1/ingest {"repo_url": "https://github.com/org/repo"}`
2. Explore the codebase to identify ~25 testable functions and classes
3. Create `tests/golden_qa/{repo_name}_qa.json` with entries following the schema above
4. Run `python tests/eval_retrieval.py --qa tests/golden_qa/{repo_name}_qa.json --repo-id <id>`
5. Iterate on chunk sizes or BM25 weights if Recall@5 < 0.70
