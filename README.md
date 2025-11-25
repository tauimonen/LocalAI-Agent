[![Project Status: Work in Progress](https://img.shields.io/badge/Project%20Status-Work%20in%20Progress-orange.svg)](https://img.shields.io/badge/Project%20Status-Work%20in%20Progress-orange.svg)

# Project: Local AI Agent with RAG, Ollama, and Python

This project provides a fully local AI development environment using Ollama, OpenWebUI, and a Python-based RAG (Retrieval-Augmented Generation) and agent framework.
The goal is to enable privacy-preserving, offline-capable AI workflows that support:

- Local large language models (LLMs)
- Document ingestion and indexing
- Vector search
- Intelligent multimodule agents
- Custom backends and tools
- Extendable UI (OpenWebUI)

Everything runs locally on Linux/Ubuntu or WSL.

## Features

### Local LLMs via Ollama
Run high-quality models (e.g., Llama 3, Mistral, Phi-3, Gemma) entirely offline.

### Custom RAG pipeline
Build document indices, parse PDFs, HTML, Markdown, and perform semantic search.

### Agent framework
Agents can call functions, use external tools, fetch documents, or query vector stores.

### Python backend
FastAPI server for custom endpoints, integration layers, and agent actions.

### Frontend GUI
Optional OpenWebUI integration to interact with models through a browser.

## Tech Stack Overview

### Local Model Layer

**Ollama**  
Core local LLM runtime and model orchestrator. Provides an HTTP API and CLI for text generation, embeddings, and chat.

### User Interface

**OpenWebUI**  
A modern web UI for interacting with Ollama models. Supports RAG plugins, chat history, file uploads, and extensions.

### Python Libraries

#### 1. RAG & LLM Frameworks
- **LlamaIndex** – High-level framework for building RAG pipelines, document loaders, query engines, and agents.  
- **LangChain** – Toolkit for chains, agents, tools, and integration with vector stores.

#### 2. Vector Databases
- **ChromaDB** – Simple, embedded vector database for fast similarity search.  
- **LanceDB** – High-performance, disk-based vector store built on Apache Arrow.

#### 3. Embedding & Tokenization
- **sentence-transformers** – Local embedding creation for semantic search.  
- **tiktoken** – Token counting and efficient text splitting.

#### 4. Document Processing
- **pymupdf** – High-accuracy PDF text extraction.  
- **pdfplumber** – Structural PDF extraction.  
- **beautifulsoup4** – HTML parsing and cleanup.  
- **markdown-it-py** – Markdown conversion.

#### 5. Networking & APIs
- **requests** – Simple synchronous HTTP client.  
- **httpx** – Async-first HTTP client.  
- **websocket-client** – Real-time streaming connection support (useful for model output).

#### 6. Backend Framework
- **FastAPI** – Lightweight, fast web backend for tool execution and agent endpoints.  
- **uvicorn** – ASGI server for running FastAPI applications.

#### 7. Configuration
- **python-dotenv** – Environment configuration via .env files.  
- **pydantic** – Data validation and structured model definitions.

---

## Project Goals

- Build a fully local AI agent system that requires no cloud services.  
- Process and index documents (PDF, HTML, Markdown) for knowledge retrieval.  
- Develop a modular RAG pipeline for production-grade retrieval accuracy.  
- Create intelligent agents capable of calling tools, running chains, or interacting with a backend.  
- Provide a simple UI via OpenWebUI while supporting programmatic interaction through Python.  
- Enable reproducible development using a virtual environment and clear dependency management.

