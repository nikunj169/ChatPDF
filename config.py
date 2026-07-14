"""Configuration settings for the ChatPDF RAG pipeline."""

CHUNKING_STRATEGIES = {
    "fixed_size": {
        "name": "Fixed-Size Chunking",
        "description": "Splits text into chunks of a fixed character length with overlap.",
        "default_chunk_size": 1000,
        "default_chunk_overlap": 200,
        "default_separator": "\n",
    },
    "semantic": {
        "name": "Semantic Chunking",
        "description": "Splits text at semantically meaningful boundaries using sentence embeddings.",
        "default_buffer_size": 3,
        "default_breakpoint_percentile": 85,
    },
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

LLM_TEMPERATURE = 0.2
RETRIEVER_K = 4
