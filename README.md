# Local RAG Chat Assistant

A local retrieval-augmented generation (RAG) chatbot that answers questions from your own documents using Llama 3.2 via Ollama.

## Problem

- Generic LLMs answer from their training data, not from your own notes or documents.
- For personal notes, capstone reports, or internal docs, you want answers grounded in **your source of truth**.
- This project provides a local, privacy-preserving RAG system for that.

## Solution

- Ingests a text document (e.g., `data/notes.txt`), splits it into chunks, and stores embeddings in a vector database (Chroma).
- On each question, retrieves the most relevant chunks.
- Sends the retrieved context + question to Llama 3.2 via Ollama.
- Returns answers grounded in your document, with optional visibility into which chunks were used.

All processing runs locally; no cloud APIs or external services are required.

## Tech Stack

- Python
- LangChain (LangChain Ollama, LangChain Text Splitters, LangChain Community)
- Ollama (Llama 3.2 for chat, `nomic-embed-text` for embeddings)
- Chroma (vector store)

## Project Structure

```text
project-root/
  data/
    notes.txt           # Your document database
  src/
    chat.py             # Plain LLM chat (no RAG)
    rag_chat.py         # Main RAG chat app
    eval_rag.py         # Simple evaluation script
  sandbox/
    # Experimental / debugging scripts (not part of the main app)
```

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
   python rag_chat.py
   ```

   Ask questions about your notes, e.g.:

   - “What is RAG?”
   - “What is this notes file mainly about?”

4. (Optional) Run the plain chat without RAG:

   ```bash
   python chat.py
   ```

5. (Optional) Run the evaluation script:

   ```bash
   python eval_rag.py
   ```

   It will run a small set of predefined questions and print PASS/FAIL for each.

## Features

- Retrieve answers from your own notes / documents.
- Configurable chunk size and overlap.
- Transparent: shows which document chunks were used for each answer.
- Fully local: no external API keys or cloud services.

## Future Work

- Add support for PDF / Markdown files.
- Add a web UI or FastAPI endpoint.
- Add more comprehensive evaluation metrics.
- Add multi-document support with metadata (e.g., by filename).