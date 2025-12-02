"""FastAPI application with RAG endpoints.

Provides REST API for document management and RAG queries.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from ai_agent.config import Settings, get_settings
from ai_agent.core.llm import OllamaClient
from ai_agent.core.vector_store import ChromaVectorStore, LanceVectorStore
from ai_agent.agents.rag_agent import RAGAgent
from ai_agent.rag.document_loader import DocumentLoader


# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    query: str = Field(..., description="User query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    use_history: bool = Field(default=False, description="Use conversation history")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""

    answer: str
    sources: list[dict[str, str | float]]
    retrieved_count: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    ollama_available: bool
    vector_db_type: str


class UploadResponse(BaseModel):
    """Document upload response."""

    filename: str
    status: str
    document_count: int


# Global state
app_state: dict[str, any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.
    Initializes and cleans up resources.
    """
    settings = get_settings()

    # Initialize components
    logger.info("Initializing application...")

    # Create Ollama client
    llm = OllamaClient(settings)

    # Check Ollama health
    if not await llm.health_check():
        logger.error("Ollama service is not available!")
        # Continue anyway for other endpoints

    # Initialize vector store
    if settings.vector_db_type == "chroma":
        vector_store = ChromaVectorStore(settings)
    else:
        vector_store = LanceVectorStore(settings)

    # Create RAG agent
    rag_agent = RAGAgent(llm, vector_store, settings)

    # Store in app state
    app_state["llm"] = llm
    app_state["vector_store"] = vector_store
    app_state["rag_agent"] = rag_agent
    app_state["document_loader"] = DocumentLoader()
    app_state["settings"] = settings

    logger.info("Application initialized successfully")

    yield

    # Cleanup
    logger.info("Shutting down application...")
    await llm.close()


# Create FastAPI app
app = FastAPI(
    title="Local AI Agent API",
    description="RAG-powered AI agent with Ollama and vector search",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict[str, str])
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Local AI Agent API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health."""
    llm: OllamaClient = app_state["llm"]
    settings: Settings = app_state["settings"]

    ollama_available = await llm.health_check()

    return HealthResponse(
        status="healthy" if ollama_available else "degraded",
        ollama_available=ollama_available,
        vector_db_type=settings.vector_db_type,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Upload and index a document.
    Args:
        file: Document file to upload
    Returns:
        Upload status and document count
    """
    settings: Settings = app_state["settings"]
    document_loader: DocumentLoader = app_state["document_loader"]
    vector_store = app_state["vector_store"]

    # Save uploaded file
    file_path = settings.uploads_dir / file.filename

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Saved uploaded file: {file.filename}")

        # Load and parse document
        document = document_loader.load(file_path)

        # Split into chunks (simple sentence splitting for now)
        chunks = []
        sentences = document.content.split(". ")

        for i, sentence in enumerate(sentences):
            if sentence.strip():
                chunks.append(
                    {
                        "id": f"{file.filename}_{i}",
                        "content": sentence.strip() + ".",
                        "metadata": {
                            **document.metadata,
                            "chunk_id": i,
                        },
                    }
                )

        # Add to vector store
        await vector_store.add_documents(chunks)

        return UploadResponse(
            filename=file.filename,
            status="success",
            document_count=len(chunks),
        )

    except Exception as e:
        logger.error(f"Failed to process upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """Query the RAG system.
    Args:
        request: Query request
    Returns:
        Query response with answer and sources
    """
    rag_agent: RAGAgent = app_state["rag_agent"]

    try:
        result = await rag_agent.query(
            query=request.query,
            top_k=request.top_k,
            use_history=request.use_history,
            temperature=request.temperature,
        )

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            retrieved_count=len(result["retrieved_documents"]),
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_rag_stream(request: QueryRequest):
    """Stream RAG query response.

    Args:
        request: Query request

    Returns:
        Streaming response
    """
    rag_agent: RAGAgent = app_state["rag_agent"]

    async def generate():
        try:
            async for chunk in rag_agent.stream_query(
                query=request.query,
                top_k=request.top_k,
                temperature=request.temperature,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Stream query failed: {e}")
            yield f"Error: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/history/clear")
async def clear_history() -> dict[str, str]:
    """Clear conversation history."""
    rag_agent: RAGAgent = app_state["rag_agent"]
    rag_agent.clear_history()
    return {"status": "history cleared"}


@app.get("/models")
async def list_models() -> dict[str, list[dict]]:
    """List available Ollama models."""
    llm: OllamaClient = app_state["llm"]

    try:
        models = await llm.list_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main() -> None:
    """Run the FastAPI application."""
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "ai_agent.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
