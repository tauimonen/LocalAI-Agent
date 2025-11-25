"""
Vector store abstraction layer provides a unified interface for ChromaDB and LanceDB with automatic
embedding generation and similarity search.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol

import chromadb
import lancedb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from sentence_transformers import SentenceTransformer

from ai_agent.config import Settings


class EmbeddingFunction(Protocol):
    """Protocol for embedding functions."""
    
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to embeddings."""
        ...


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    async def add_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        """Add documents to the vector store.
        Args:
            documents: List of document dicts with 'content' and 'metadata'
            embeddings: Optional pre-computed embeddings
        Returns:
            List of document IDs
        """
        ...
    
    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents.
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Metadata filters
        Returns:
            List of matching documents with scores
        """
        ...
    
    @abstractmethod
    async def delete_collection(self) -> None:
        """Delete the entire collection."""
        ...


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation."""
    
    def __init__(self, settings: Settings, collection_name: str = "documents") -> None:
        """Initialize ChromaDB vector store.
        Args:
            settings: Application settings
            collection_name: Name of the collection
        """
        self.settings = settings
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
        logger.info(f"Initialized ChromaDB collection: {collection_name}")
    
    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts.
        Args:
            texts: List of text strings
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
    
    async def add_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        """Add documents to ChromaDB.
        Args:
            documents: List of document dicts
            embeddings: Optional pre-computed embeddings
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Extract content and metadata
        ids = [doc.get("id", f"doc_{i}") for i, doc in enumerate(documents)]
        contents = [doc["content"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        
        # Generate embeddings if not provided
        if embeddings is None:
            logger.debug(f"Generating embeddings for {len(documents)} documents")
            embeddings = self._generate_embeddings(contents)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
        return ids
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents in ChromaDB.
        Args:
            query: Search query
            top_k: Number of results
            filter_dict: Metadata filters
        Returns:
            List of matching documents
        """
        # Generate query embedding
        query_embedding = self._generate_embeddings([query])[0]
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict,
        )
        
        # Format results
        documents = []
        for i in range(len(results["ids"][0])):
            documents.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - results["distances"][0][i],  # Convert distance to similarity
            })
        
        logger.debug(f"Found {len(documents)} documents for query")
        return documents
    
    async def delete_collection(self) -> None:
        """Delete the ChromaDB collection."""
        self.client.delete_collection(name=self.collection_name)
        logger.info(f"Deleted ChromaDB collection: {self.collection_name}")


class LanceVectorStore(VectorStore):
    """LanceDB vector store implementation."""
    
    def __init__(self, settings: Settings, table_name: str = "documents") -> None:
        """Initialize LanceDB vector store.
        Args:
            settings: Application settings
            table_name: Name of the table
        """
        self.settings = settings
        self.table_name = table_name
        
        # Initialize LanceDB
        self.db = lancedb.connect(str(settings.lance_db_path))
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        
        logger.info(f"Initialized LanceDB at: {settings.lance_db_path}")
    
    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
    
    async def add_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        """Add documents to LanceDB."""
        if not documents:
            return []
        
        # Generate embeddings if not provided
        contents = [doc["content"] for doc in documents]
        if embeddings is None:
            embeddings = self._generate_embeddings(contents)
        
        # Prepare data
        data = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            data.append({
                "id": doc.get("id", f"doc_{i}"),
                "content": doc["content"],
                "vector": embedding,
                "metadata": str(doc.get("metadata", {})),
            })
        
        # Create or append to table
        if self.table_name in self.db.table_names():
            table = self.db.open_table(self.table_name)
            table.add(data)
        else:
            self.db.create_table(self.table_name, data)
        
        logger.info(f"Added {len(documents)} documents to LanceDB")
        return [d["id"] for d in data]
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents in LanceDB."""
        # Generate query embedding
        query_embedding = self._generate_embeddings([query])[0]
        
        # Open table and search
        table = self.db.open_table(self.table_name)
        results = table.search(query_embedding).limit(top_k).to_list()
        
        # Format results
        documents = []
        for result in results:
            documents.append({
                "id": result["id"],
                "content": result["content"],
                "metadata": eval(result["metadata"]),  # Convert string back to dict
                "score": float(result["_distance"]),
            })
        
        return documents
    
    async def delete_collection(self) -> None:
        """Delete the LanceDB table."""
        self.db.drop_table(self.table_name)
        logger.info(f"Deleted LanceDB table: {self.table_name}")