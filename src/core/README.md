# Vector Store Layer for AI Agent

This module provides a **vector store abstraction layer**, allowing the AI agent to store and retrieve documents using vector embeddings. It supports multiple backend databases with a unified interface.

---

## Key Features

- **Unified interface** for multiple vector databases:
  - `ChromaVectorStore` (ChromaDB)
  - `LanceVectorStore` (LanceDB)
- Automatic **embedding generation** using `SentenceTransformer`
- Async methods for efficient integration with web services (FastAPI)
- Handles document metadata and similarity search
- Supports RAG (Retrieval-Augmented Generation) workflows

---

## Main Components

### 1. `EmbeddingFunction` (Protocol)
Defines the interface for embedding functions:
- `encode(texts: list[str]) -> list[list[float]]`
- Ensures consistent embedding creation.

### 2. `VectorStore` (Abstract Base Class)
Defines the abstract methods for all vector stores:
- `add_documents(documents, embeddings)` → Add documents
- `search(query, top_k, filter_dict)` → Retrieve similar documents
- `delete_collection()` → Remove all stored documents

### 3. `ChromaVectorStore`
- Implements `VectorStore` for **ChromaDB**
- Initializes client, collection, and embedding model
- Handles document addition, search, and collection deletion
- Converts distances to similarity scores for queries

### 4. `LanceVectorStore`
- Implements `VectorStore` for **LanceDB**
- Initializes database and embedding model
- Adds documents with embeddings
- Performs similarity search
- Deletes entire table when needed

---

## Why This Module is Important

- Provides a **consistent abstraction** over different vector databases
- Simplifies embedding generation and management
- Enables fast and reliable document retrieval for RAG workflows
- Supports asynchronous operations for high-performance web applications

---

## Usage Example (High-Level)

```python
from ai_agent.vector_store import ChromaVectorStore
from ai_agent.config import Settings

settings = Settings()
store = ChromaVectorStore(settings)

# Add documents
await store.add_documents([{"content": "Hello world", "metadata": {}}])

# Search
results = await store.search("Hello")

# Delete collection
await store.delete_collection()
