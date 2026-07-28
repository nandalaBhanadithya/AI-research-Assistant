---
title: AI Research Assistant
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# AI Research & Knowledge Assistant

A production-oriented RAG (Retrieval-Augmented Generation) backend for uploading technical/research PDFs, searching and asking grounded questions over them with citations, comparing and summarizing documents, and automatically classifying them.

## Deployment Configuration

This Space uses cloud embeddings (Hugging Face Inference API) and Groq for generation, eliminating the need for local Ollama.

### Required Secrets

Set these in the Space's Settings > Secrets:

- `GROQ_API_KEY`: Your Groq API key (get from https://console.groq.com)
- `HF_API_KEY`: Your Hugging Face API key (get from https://huggingface.co/settings/tokens)
- `EMBEDDING_PROVIDER`: Set to `huggingface`
- `GENERATION_PROVIDER`: Set to `groq`

### Environment Variables

The following are set by default but can be overridden:

- `HF_EMBEDDING_MODEL`: `sentence-transformers/all-MiniLM-L6-v2`
- `GROQ_MODEL`: `llama-3.1-8b-instant`

## Usage

1. Visit the Space URL
2. Access the interactive API documentation at `/docs`
3. Upload PDFs via the API endpoints
4. Search, ask questions, compare documents, and generate summaries

## Architecture

- **FastAPI** for the REST API
- **SQLite** for metadata storage
- **ChromaDB** for vector search
- **Hugging Face Inference API** for embeddings
- **Groq** for LLM generation
- **TensorFlow** for document classification

See the main README.md for detailed documentation.
