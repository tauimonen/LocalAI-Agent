# Text Splitter Module

## Overview

The Text Splitter module provides intelligent text chunking strategies for Retrieval-Augmented Generation (RAG) systems. It breaks down large documents into smaller, overlapping chunks that are optimized for embedding generation and semantic search.

## Why Text Splitting?

When working with RAG systems, documents are often too large to process as a single unit. Text splitting solves several key problems:

1. **Embedding Size Limits**: Embedding models have maximum token limits (typically 512-8192 tokens)
2. **Retrieval Precision**: Smaller chunks allow more precise matching of relevant content
3. **Context Management**: Smaller chunks fit better within LLM context windows
4. **Information Density**: Each chunk contains focused, coherent information

## Components

### 1. TextSplitter (Base Class)

The fundamental text splitter with configurable chunk size and overlap.

**Key Features:**
- Simple character-based splitting
- Sentence boundary detection
- Configurable overlap between chunks
- Two splitting modes: character-based and sentence-based

**Example:**
```python
from ai_agent.rag.text_splitter import TextSplitter

splitter = TextSplitter(chunk_size=512, chunk_overlap=50)

text = "Your long document text here..."
chunks = splitter.split(text)

# Result: List of text chunks, each ~512 characters
# with 50 characters of overlap between consecutive chunks
```

### 2. RecursiveCharacterTextSplitter

Advanced splitter that tries multiple separators hierarchically to create natural text boundaries.

**Key Features:**
- Hierarchical splitting strategy
- Tries separators in order: `\n\n` → `\n` → `. ` → ` ` → force split
- Preserves document structure (paragraphs, sentences)
- Better semantic coherence in chunks

**Example:**
```python
from ai_agent.rag.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

chunks = splitter.split(text)
```

## How It Works

### Character-Based Splitting (`split()`)

```
Original Text: "This is sentence one. This is sentence two. This is sentence three."

Parameters:
- chunk_size = 30
- chunk_overlap = 10

Result:
Chunk 1: "This is sentence one. This"
Chunk 2: "This is sentence two. This"
Chunk 3: "This is sentence three."
         ^^^^^^^^^^^ (overlap region)
```

**Algorithm:**
1. Start at position 0
2. Take next `chunk_size` characters
3. Try to break at sentence boundary (`. `)
4. Move forward by `chunk_size - chunk_overlap`
5. Repeat until end of text

### Sentence-Based Splitting (`split_by_sentences()`)

Splits on sentence boundaries and groups sentences into chunks.

```
Text: "Sentence A. Sentence B. Sentence C. Sentence D."

Parameters:
- chunk_size = 30 characters

Result:
Chunk 1: "Sentence A. Sentence B."
Chunk 2: "Sentence B. Sentence C."  (overlap includes Sentence B)
Chunk 3: "Sentence C. Sentence D."  (overlap includes Sentence C)
```

**Benefits:**
- Preserves complete sentences
- More semantic coherence
- Better for Q&A systems

### Recursive Splitting

The recursive splitter tries separators in order of priority:

```
Document Structure:

Paragraph 1

Paragraph 2

Paragraph 3

Step 1: Try splitting by "\n\n" (paragraphs)
Step 2: If paragraph too large, try "\n" (lines)
Step 3: If line too large, try ". " (sentences)
Step 4: If sentence too large, try " " (words)
Step 5: Force split if still too large
```

## Configuration Parameters

### chunk_size (int, default=512)
Target size for each chunk in characters.

**Considerations:**
- Smaller chunks (256-512): Better precision, more chunks to manage
- Larger chunks (1024-2048): More context, fewer chunks, less precise retrieval

### chunk_overlap (int, default=50)
Number of characters to overlap between consecutive chunks.

**Why Overlap Matters:**
- Prevents information loss at chunk boundaries
- Ensures context continuity
- Typical values: 10-20% of chunk_size

**Example:**
```
No Overlap:                With Overlap:
[Chunk 1: "...end"]        [Chunk 1: "...end of text"]
[Chunk 2: "Start..."]      [Chunk 2: "of text Start..."]
                                     ^^^^^^^^ (overlap)
```

### separators (List[str])
Ordered list of separators for recursive splitting.

**Default:** `["\n\n", "\n", ". ", " ", ""]`

## Usage Examples

### Basic Usage

```python
from ai_agent.rag.text_splitter import TextSplitter

# Create splitter
splitter = TextSplitter(chunk_size=512, chunk_overlap=50)

# Split text
document = """
This is a long document with multiple paragraphs.

Each paragraph contains several sentences.
We want to split this intelligently.
"""

chunks = splitter.split(document)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk[:50]}...")
```

### Sentence-Based Splitting

```python
splitter = TextSplitter(chunk_size=512, chunk_overlap=50)

# Better for preserving sentence structure
chunks = splitter.split_by_sentences(document)
```

### Recursive Splitting (Recommended)

```python
from ai_agent.rag.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " "]
)

chunks = splitter.split(document)
```

### Custom Separators for Code

```python
# For splitting code files
code_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\nclass ", "\n\ndef ", "\n\n", "\n", " "]
)

code_chunks = code_splitter.split(python_code)
```

## Best Practices

### 1. Choose Appropriate Chunk Size

```python
# For Q&A systems (precision)
qa_splitter = TextSplitter(chunk_size=256, chunk_overlap=25)

# For summarization (context)
summary_splitter = TextSplitter(chunk_size=1500, chunk_overlap=150)

# General purpose
default_splitter = TextSplitter(chunk_size=512, chunk_overlap=50)
```

### 2. Use Recursive Splitter for Documents

```python
# Preserves document structure
doc_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
```

### 3. Adjust Overlap Based on Use Case

```python
# High overlap for critical information
critical_splitter = TextSplitter(chunk_size=500, chunk_overlap=100)  # 20%

# Low overlap for large datasets
efficiency_splitter = TextSplitter(chunk_size=1000, chunk_overlap=50)  # 5%
```

### 4. Consider Token Limits

```python
# Approximate tokens = characters / 4
# For 512 token limit embedding model:
splitter = TextSplitter(
    chunk_size=2048,  # ~512 tokens
    chunk_overlap=200  # ~50 tokens
)
```

## Integration with RAG Pipeline

```python
from ai_agent.rag.document_loader import DocumentLoader
from ai_agent.rag.text_splitter import RecursiveCharacterTextSplitter
from ai_agent.core.vector_store import ChromaVectorStore

# 1. Load document
loader = DocumentLoader()
document = loader.load("document.pdf")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
chunks = splitter.split(document.content)

# 3. Create chunk documents
chunk_docs = []
for i, chunk in enumerate(chunks):
    chunk_docs.append({
        "id": f"{document.metadata['filename']}_{i}",
        "content": chunk,
        "metadata": {
            **document.metadata,
            "chunk_id": i,
            "total_chunks": len(chunks)
        }
    })

# 4. Add to vector store
await vector_store.add_documents(chunk_docs)
```

## Performance Considerations

### Memory Usage
- Holds entire text in memory
- O(n) space complexity where n = text length
- For very large documents (>10MB), consider streaming

### Processing Speed
- Character splitting: O(n)
- Sentence splitting: O(n) with regex overhead
- Recursive splitting: O(n × m) where m = number of separators

### Optimization Tips
```python
# Process large documents in batches
def split_large_document(text: str, batch_size: int = 100000):
    splitter = TextSplitter()
    
    # Split into manageable sections
    sections = [text[i:i+batch_size] for i in range(0, len(text), batch_size)]
    
    all_chunks = []
    for section in sections:
        all_chunks.extend(splitter.split(section))
    
    return all_chunks
```

## Common Issues and Solutions

### Issue: Chunks Too Large for Embedding Model

```python
# Solution: Reduce chunk_size
splitter = TextSplitter(chunk_size=256, chunk_overlap=25)
```

### Issue: Context Loss at Boundaries

```python
# Solution: Increase overlap
splitter = TextSplitter(chunk_size=512, chunk_overlap=100)  # 20% overlap
```

### Issue: Poor Semantic Coherence

```python
# Solution: Use RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". "]
)
```

## Testing

```python
def test_text_splitter():
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    
    text = "A" * 250  # 250 characters
    chunks = splitter.split(text)
    
    # Verify chunk sizes
    assert all(len(chunk) <= 100 for chunk in chunks)
    
    # Verify overlap
    for i in range(len(chunks) - 1):
        overlap = chunks[i][-20:]
        assert chunks[i+1].startswith(overlap)
```

## API Reference

### TextSplitter

#### Constructor
```python
TextSplitter(chunk_size: int = 512, chunk_overlap: int = 50)
```

#### Methods

**split(text: str) -> List[str]**
- Character-based splitting with sentence boundary detection
- Returns list of text chunks

**split_by_sentences(text: str) -> List[str]**
- Sentence-aware splitting
- Preserves complete sentences
- Returns list of chunks containing full sentences

### RecursiveCharacterTextSplitter

#### Constructor
```python
RecursiveCharacterTextSplitter(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None
)
```

#### Methods

**split(text: str) -> List[str]**
- Recursive hierarchical splitting
- Tries separators in order
- Returns optimally split chunks

## Related Components

- **DocumentLoader**: Loads documents from files → feeds into TextSplitter
- **DocumentIndexer**: Uses TextSplitter to prepare documents for indexing
- **VectorStore**: Stores chunks created by TextSplitter
- **RAGAgent**: Retrieves and uses chunks for generation

## Further Reading

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/)
- [Semantic Chunking with LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)
