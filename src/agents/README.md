# RAGAgent (Retrieval-Augmented Generation Agent)

`RAGAgent` is a core class for a **retrieval-augmented generation (RAG) workflow**.  
It combines **document retrieval** with **LLM generation** to provide context-aware AI responses.

---

## Components

- **vector_store : VectorStore**  
  Abstract interface to ChromaDB or LanceDB. Retrieves top-k relevant documents.

- **llm : OllamaClient**  
  Interface to Ollama LLM (e.g., Llama 3, Mistral, Qwen). Supports:
  - `.generate(prompt)`
  - `.chat(messages)`
  - `.stream_chat(messages)`

- **settings : Settings**  
  Configuration for max retrieval results, similarity threshold, embedding models, etc.

- **conversation_history**  
  Stores past user-assistant messages for multi-turn conversations.

---

## Key Methods

### 1. `_build_context_prompt(query, retrieved_docs)`
- Builds a single, formatted prompt combining:
  - user query
  - retrieved documents
  - document metadata and relevance scores

### 2. `async query(query, top_k, use_history, temperature)`
Main function to process a user query:

1. Retrieve documents from the vector store.
2. Filter by similarity threshold.
3. Build context prompt with `_build_context_prompt`.
4. Send prompt to LLM:
   - **Single-turn:** `generate(prompt)`
   - **Multi-turn (with history):** `chat(messages)`
5. Return structured output:
   - `answer`
   - `retrieved_documents`
   - `sources` (metadata + relevance)

---

### 3. `async stream_query(query, top_k, temperature)`
- Same as `query`, but streams response chunks.  
- Ideal for real-time frontend interfaces or websockets.

---

### 4. `clear_history()` / `get_history()`
- `clear_history()`: clears conversation history.
- `get_history()`: returns a copy of conversation history.

---

## Purpose

The `RAGAgent` is the **core of a local AI agent**:

- Combines vector store retrieval + LLM generation
- Formats and injects context into prompts
- Supports conversation history for multi-turn chats
- Extracts and tracks document sources and metadata
- Supports streaming responses for responsive UIs

---

## Summary

- Retrieves relevant documents
- Filters by similarity threshold
- Builds context-aware prompts
- Sends prompts to an LLM
- Returns:
  - AI-generated answer
  - Retrieved documents
  - Source metadata
- Supports streaming and multi-turn conversation


