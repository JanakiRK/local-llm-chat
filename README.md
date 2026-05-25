# Local RAG Chat Assistant

A local retrieval-augmented generation (RAG) chatbot that answers questions from your own documents using Llama 3.2 via Ollama.

## Problem

- Generic LLMs answer from their training data, not from your own notes or documents.
- For personal notes, capstone reports, or internal docs, you want answers grounded in **your source of truth**.
- This project provides a local, privacy-preserving RAG system for that.

## Solution

- Ingests multiple documents from the `data/` folder (e.g. `.txt` and `.md` files), splits them into chunks, and stores embeddings with metadata (including source filename) in a vector database (Chroma).
- On each question, retrieves the most relevant chunks.
- Sends the retrieved context + question to Llama 3.2 via Ollama.
- Returns answers grounded in your documents, with visibility into which files and chunks were used.

All processing runs locally; no cloud APIs or external services are required.

## Tech Stack

- Python
- LangChain (LangChain Ollama, LangChain Text Splitters)
- Ollama (Llama 3.2 for chat, `nomic-embed-text` for embeddings)
- Chroma (vector store, via `langchain-chroma`)

## Project Structure

```text
project-root/
  data/
    notes_rag.txt        # RAG and LLM notes
    notes_python.txt     # Python notes
    project_ideas.md     # Project ideas (Markdown)
  src/
    chat.py              # Plain LLM chat (no RAG)
    rag.py               # Main RAG chat app (multi-document)
    eval_rag.py          # Simple evaluation script
  sandbox/
    # Experimental / debugging scripts (not part of the main app)
```

_(If your main file is still named `rag_chat.py`, replace `rag.py` above with `rag_chat.py`.)_

## How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Make sure Ollama is running and the models are available:

   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

3. Run the RAG chat (main app):

   ```bash
   cd src
   python rag.py
   # or: python rag_chat.py  (if that's your filename)
   ```

   Ask questions about your documents, e.g.:

   - “What is Retrieval-Augmented Generation?”
   - “What is a Python list comprehension?”
   - “Give me some project ideas.”

4. (Optional) Run the plain chat without RAG:

   ```bash
   cd src
   python chat.py
   ```

5. (Optional) Run the evaluation script:

   ```bash
   cd src
   python eval_rag.py
   ```

   It will run a small set of predefined questions and print PASS/FAIL for each.

## Features

- Retrieve answers from your own notes / documents.
- Support for multiple documents with per-chunk `source` metadata (e.g. filename).
- Configurable chunk size and overlap.
- Transparent: shows which document chunks and files were used for each answer.
- Fully local: no external API keys or cloud services.

## Future Work

- Add support for PDF files.
- Add a web UI or FastAPI endpoint.
- Add more comprehensive evaluation metrics.