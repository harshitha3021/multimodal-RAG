# Enterprise Multimodal RAG Assistant

**Enterprise-grade Multimodal Retrieval-Augmented Generation (RAG) system** built with Streamlit, Groq LLM, Jina embeddings, FAISS, and EasyOCR.  
Supports **text and image inputs** and performs **semantic retrieval and generation** for knowledge-based applications.

---

## 🚀 Features

- **Multimodal Input**: Upload PDF, TXT, JPG, PNG images.
- **Text Chunking**: Sentence-aware chunking with configurable size and overlap.
- **Embeddings**: Jina embeddings (batched, normalized, with retry logic).
- **Vector Search**: FAISS cosine similarity retriever with metadata filtering.
- **Reranking**: Keyword-based and embedding-based reranker for improved results.
- **LLM Answer Generation**: Groq LLM (LLAMA-3 / LLAMA-4) with context grounding.
- **Vision Module**: Describe images and extract text using EasyOCR + Groq Vision.
- **Session Management**: Chat history saved per session with performance metrics.
- **Enterprise Safety**: Handles large documents/images, timeout/retry logic, and error handling.

---

## 🗂 Project Structure

multimodal-rag/
│
├─ app.py # Main Streamlit app
├─ chunking.py # Sentence-aware text chunking
├─ embeddings.py # Jina embeddings with batching & retry
├─ retriever.py # FAISS cosine similarity retriever
├─ reranker.py # Keyword/embedding reranker
├─ llm.py # Groq LLM interface with retry logic
├─ vision.py # Image description using Groq Vision
├─ ocr.py # OCR using EasyOCR
├─ config.py # Centralized config and hyperparameters
├─ requirements.txt # Python dependencies
└─ README.md # Project documentation