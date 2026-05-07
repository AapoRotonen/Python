# AI Knowledge Base Agent (RAG)

A Python-based AI application leveraging **RAG (Retrieval-Augmented Generation)** architecture to analyze PDF documents and answer questions about them accurately — without hallucinations.

---

## Features

- **Document analysis** — Loads and processes PDF files using `PyPDFLoader`
- **Semantic search** — Uses OpenAI Embeddings to understand the meaning of text, not just keywords
- **Vector database** — ChromaDB for efficient document storage and retrieval
- **Fact-based answers** — Instructs GPT-4o to answer only from the provided context
- **Conversation memory** — Maintains chat history across turns using `RunnableWithMessageHistory`
- **Terminal chat** — Lightweight CLI interface for quick Q&A sessions
- **Web UI** — Streamlit-powered chat interface with session management
- **Test suite** — Pytest-based tests covering unit, integration, edge cases, and UI logic

---

## Project Structure

```
AI/
├── app.py                  # Core RAG logic (LangChain + ChromaDB + GPT-4o)
├── chat.py                 # Terminal chat interface with conversation memory
├── streamlit_app.py        # Web chat UI built with Streamlit
├── conftest.py             # Shared pytest fixtures (root level)
├── .env                    # API keys (not committed to version control)
├── Rotonen_Aapo_CV_2026_Full.pdf
└── tests/
    ├── __init__.py
    ├── conftest.py         # Test-level fixture placeholder
    ├── test_app.py         # Tests for core RAG logic
    ├── test_chat.py        # Tests for terminal chat loop
    └── test_streamlit_app.py  # Tests for Streamlit UI logic
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12+ |
| Orchestration | LangChain (LCEL) |
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI Embeddings |
| Vector Store | ChromaDB |
| Web UI | Streamlit |
| Testing | Pytest, unittest.mock, Streamlit AppTest |

---

## Installation

**1. Clone the repo and create a virtual environment:**

```bash
python -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate       # macOS / Linux
```

**2. Install dependencies:**

```bash
pip install langchain langchain-openai langchain-community langchain-text-splitters \
            chromadb pypdf python-dotenv openai streamlit pytest
```

**3. Configure environment variables:**

Create a `.env` file in the `AI/` folder:

```
OPENAI_API_KEY=your_api_key_here
```

---

## Usage

### Terminal chat

```bash
py chat.py
```

Type your questions and press Enter. Type `lopeta`, `exit`, or `quit` to stop.

### Web UI (Streamlit)

```bash
py -m streamlit run streamlit_app.py
```

Opens automatically at `http://localhost:8501`.

### Run core RAG logic directly

```bash
py app.py
```

---

## Running Tests

```bash
# All tests
py -m pytest tests/ -v

# Individual test files
py -m pytest tests/test_app.py -v        # RAG core logic
py -m pytest tests/test_chat.py -v       # Terminal chat loop
py -m pytest tests/test_streamlit_app.py -v  # Streamlit UI
```

### Test coverage

| File | What's tested |
|---|---|
| `test_app.py` | `retrieve_context`, session store, text splitter boundary values, document metadata, edge cases, end-to-end with mocked LLM |
| `test_chat.py` | Exit commands (`lopeta`/`exit`/`quit`), empty input skipping, chain invocation parameters, session ID consistency, stdout output |
| `test_streamlit_app.py` | Session state initialization, send logic, input validation, message ordering, cache behavior, Streamlit AppTest UI simulation |

All tests use `unittest.mock` — no real OpenAI API calls are made during testing.

---

## How It Works

```
PDF Document
    │
    ▼
PyPDFLoader → CharacterTextSplitter → OpenAI Embeddings → ChromaDB
                                                               │
User Question ──────────────────────────────────────► Semantic Search
                                                               │
                                                        Relevant Chunks
                                                               │
                                              Chat History + System Prompt
                                                               │
                                                           GPT-4o
                                                               │
                                                            Answer
```