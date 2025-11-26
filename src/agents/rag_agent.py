"""
RAG Agent with retrieval-augmented generation.

This agent retrieves relevant documents from the vector store
and uses them to augment LLM responses.
"""

from typing import Any

from loguru import logger

from ai_agent.config import Settings
from ai_agent.core.llm import OllamaClient
from ai_agent.core.vector_store import VectorStore


class RAGAgent:
    """Retrieval-Augmented Generation agent.
    
    Combines document retrieval with LLM generation to provide
    context-aware responses.
    
    Attributes:
        llm: Ollama LLM client
        vector_store: Vector database for document retrieval
        settings: Application settings
    """
    
    def __init__(
        self,
        llm: OllamaClient,
        vector_store: VectorStore,
        settings: Settings,
    ) -> None:
        """Initialize RAG agent.
        
        Args:
            llm: Language model client
            vector_store: Vector store for retrieval
            settings: Application settings
        """
        self.llm = llm
        self.vector_store = vector_store
        self.settings = settings
        self.conversation_history: list[dict[str, str]] = []
    
    def _build_context_prompt(
        self,
        query: str,
        retrieved_docs: list[dict[str, Any]],
    ) -> str:
        """Build a prompt with retrieved context.
        Args:
            query: User query
            retrieved_docs: Retrieved documents
        Returns:
            Formatted prompt with context
        """
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.get("metadata", {}).get("source", "Unknown")
            score = doc.get("score", 0.0)
            content = doc.get("content", "")
            
            context_parts.append(
                f"[Document {i}] (Source: {source}, Relevance: {score:.2f})\n{content}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""You are a helpful AI assistant. Use the following context documents to answer the user's question. If the answer cannot be found in the context, say so clearly.

Context Documents:
{context}

---

User Question: {query}

Answer:"""
        
        return prompt
    
    async def query(
        self,
        query: str,
        top_k: int | None = None,
        use_history: bool = False,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Process a query using RAG.
        Args:
            query: User query
            top_k: Number of documents to retrieve
            use_history: Whether to use conversation history
            temperature: LLM temperature
        Returns:
            Response dict with answer and retrieved documents
        """
        logger.info(f"Processing RAG query: {query[:100]}...")
        
        # Retrieve relevant documents
        top_k = top_k or self.settings.max_retrieval_results
        retrieved_docs = await self.vector_store.search(query, top_k=top_k)
        
        if not retrieved_docs:
            logger.warning("No relevant documents found")
            response = await self.llm.generate(
                prompt=query,
                temperature=temperature,
            )
            return {
                "answer": response,
                "retrieved_documents": [],
                "sources": [],
            }
        
        # Filter by similarity threshold
        filtered_docs = [
            doc for doc in retrieved_docs
            if doc.get("score", 0) >= self.settings.similarity_threshold
        ]
        
        if not filtered_docs:
            logger.warning("No documents met similarity threshold")
            filtered_docs = retrieved_docs[:1]  # Use top result anyway
        
        # Build prompt with context
        prompt = self._build_context_prompt(query, filtered_docs)
        
        # Generate response
        if use_history and self.conversation_history:
            # Use chat mode with history
            messages = self.conversation_history + [
                {"role": "user", "content": prompt}
            ]
            response = await self.llm.chat(messages, temperature=temperature)
            
            # Update history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": response})
        else:
            # Single-turn generation
            response = await self.llm.generate(prompt, temperature=temperature)
        
        # Extract sources
        sources = [
            {
                "source": doc.get("metadata", {}).get("source", "Unknown"),
                "filename": doc.get("metadata", {}).get("filename", "Unknown"),
                "score": doc.get("score", 0.0),
            }
            for doc in filtered_docs
        ]
        
        logger.info(f"Generated response with {len(sources)} sources")
        
        return {
            "answer": response,
            "retrieved_documents": filtered_docs,
            "sources": sources,
        }
    
    async def stream_query(
        self,
        query: str,
        top_k: int | None = None,
        temperature: float = 0.7,
    ):
        """Stream RAG query response.
        Args:
            query: User query
            top_k: Number of documents to retrieve
            temperature: LLM temperature
        Yields:
            Response chunks
        """
        # Retrieve documents
        top_k = top_k or self.settings.max_retrieval_results
        retrieved_docs = await self.vector_store.search(query, top_k=top_k)
        
        if not retrieved_docs:
            async for chunk in self.llm.stream_chat(
                [{"role": "user", "content": query}],
                temperature=temperature,
            ):
                yield chunk
            return
        
        # Build prompt and stream
        prompt = self._build_context_prompt(query, retrieved_docs)
        messages = [{"role": "user", "content": prompt}]
        
        async for chunk in self.llm.stream_chat(messages, temperature=temperature):
            yield chunk
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        
        self.conversation_history.clear()
        logger.debug("Cleared conversation history")
    
    def get_history(self) -> list[dict[str, str]]:
        """Get conversation history.
        
        Returns:
            List of message dicts
        """
        return self.conversation_history.copy()