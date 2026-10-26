# Architecture

```text
PDF files in data/raw/
        |
        v
PyMuPDF: extract text + page metadata
        |
        v
Chunker: overlapping text passages
        |
        v
Sentence Transformers: embeddings
        |
        v
ChromaDB: persistent vector index

User question
        |
        v
Embedding model -> ChromaDB similarity search -> top-k passages
        |
        v
Prompt builder -> Ollama or OpenAI-compatible LLM
        |
        v
Answer + source file/page references
        |
        v
Streamlit UI or FastAPI endpoint
```

## Important boundary
Retrieval returns candidate passages; it does not prove that the final answer is correct. Verify citations and evaluate the system with manually checked questions.
