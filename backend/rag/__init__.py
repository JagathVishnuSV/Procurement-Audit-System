"""
backend/rag/
──────────────────────────────────────────────────────────────────────────────
Sprint 3: Smart CLM & RAG Pipeline

Modules
-------
embedder     – Singleton SentenceTransformer wrapper (all-MiniLM-L6-v2)
chunker      – PDF / text → semantic clause chunks via LangChain
vector_store – FAISS index CRUD (add, search, persist, load)
clm_service  – Upload orchestrator: PDF → chunks → FAISS + PostgreSQL
"""
