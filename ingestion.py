"""Ingestion pipeline: PDF extraction, preprocessing, and configurable chunking."""

import re
from typing import List, Dict

import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import EMBEDDING_MODEL, EMBEDDING_DEVICE


def get_pdf_pages(docs) -> List[Dict]:
    """Extract text from PDFs with page-level metadata."""
    pages = []
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "text": text,
                    "source": pdf.name,
                    "page": i + 1,
                })
    return pages


def preprocess_text(text: str) -> str:
    """Normalize whitespace and remove artifacts from PDF extraction."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def get_chunks_fixed_size(pages: List[Dict], chunk_size: int = 1000,
                          chunk_overlap: int = 200, separator: str = "\n") -> List[Document]:
    """Fixed-size character-based chunking with metadata preservation."""
    text_splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    documents = []
    for page in pages:
        cleaned = preprocess_text(page["text"])
        chunks = text_splitter.split_text(cleaned)
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={"source": page["source"], "page": page["page"]},
            ))
    return documents


def get_chunks_semantic(pages: List[Dict], buffer_size: int = 3,
                        breakpoint_percentile: int = 85) -> List[Document]:
    """Semantic chunking using sentence embeddings to find natural break points."""
    model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
    documents = []

    for page in pages:
        cleaned = preprocess_text(page["text"])
        sentences = _split_into_sentences(cleaned)
        if len(sentences) <= 1:
            if sentences:
                documents.append(Document(
                    page_content=sentences[0],
                    metadata={"source": page["source"], "page": page["page"]},
                ))
            continue

        sentence_embeddings = model.encode(sentences)

        similarities = []
        for i in range(len(sentences) - buffer_size):
            group_start = i
            group_end = min(i + buffer_size, len(sentences))
            group_embedding = np.mean(sentence_embeddings[group_start:group_end], axis=0).reshape(1, -1)
            next_start = i + buffer_size
            next_end = min(i + 2 * buffer_size, len(sentences))
            next_embedding = np.mean(sentence_embeddings[next_start:next_end], axis=0).reshape(1, -1)
            sim = cosine_similarity(group_embedding, next_embedding)[0][0]
            similarities.append(sim)

        if not similarities:
            documents.append(Document(
                page_content=" ".join(sentences),
                metadata={"source": page["source"], "page": page["page"]},
            ))
            continue

        threshold = np.percentile(similarities, 100 - breakpoint_percentile)

        chunks = []
        current_chunk = [sentences[0]]
        sim_idx = 0
        for i in range(1, len(sentences)):
            if sim_idx < len(similarities) and similarities[sim_idx] < threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
            if i >= buffer_size:
                sim_idx += 1
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={"source": page["source"], "page": page["page"]},
            ))

    return documents


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex patterns."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def get_chunks(pages: List[Dict], strategy: str = "fixed_size",
               chunk_size: int = 1000, chunk_overlap: int = 200,
               separator: str = "\n", buffer_size: int = 3,
               breakpoint_percentile: int = 85) -> List[Document]:
    """Route to the appropriate chunking strategy."""
    if strategy == "semantic":
        return get_chunks_semantic(pages, buffer_size, breakpoint_percentile)
    return get_chunks_fixed_size(pages, chunk_size, chunk_overlap, separator)
