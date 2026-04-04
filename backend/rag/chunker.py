"""
backend/rag/chunker.py
──────────────────────────────────────────────────────────────────────────────
PDF → semantic clause chunks.

Strategy: Semantic Clause Chunking
  1. Load PDF with LangChain's PyPDFLoader (pypdf under the hood).
  2. Split on clause-like headers (numbered sections, ALL-CAPS headings)
     using a RecursiveCharacterTextSplitter tuned for legal text.
  3. Each chunk is decorated with metadata:
       { "source": filename, "page": n, "chunk_index": i,
         "contract_id": uuid_str }

Why this chunking strategy?
- Federal contracts are structured as numbered sections.
- Splitting at section boundaries keeps clause semantics intact.
- Overlapping (chunk_overlap=50) preserves context across boundaries.

Usage
-----
    from backend.rag.chunker import chunk_pdf, chunk_text
    chunks = chunk_pdf("data/contracts/vendor_a.pdf", contract_id="abc-123")
    # → List[{"text": "...", "metadata": {...}}]
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# ── Chunking constants ────────────────────────────────────────────────────────
CHUNK_SIZE    = 600   # target characters per chunk
CHUNK_OVERLAP = 80    # overlap to preserve cross-boundary context
MIN_CHUNK_LEN = 40    # discard chunks shorter than this


def _make_splitter():
    """Build a LangChain RecursiveCharacterTextSplitter for legal text."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Legal docs: split on section markers first, then paragraphs, then words
        separators=[
            r"\n(?=\d+\.\d*\s)",   # "1.1 Pricing Terms"
            r"\n(?=[A-Z][A-Z ]{4,}\n)",  # ALL-CAPS section header
            "\n\n",
            "\n",
            ". ",
            " ",
        ],
        is_separator_regex=True,
        length_function=len,
        add_start_index=True,
    )


def chunk_text(
    text: str,
    contract_id: str,
    source_name: str = "inline",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Split a raw text string into clause-sized chunks.

    Parameters
    ----------
    text          : full contract text
    contract_id   : UUID string of the Contract DB row (for FAISS payloads)
    source_name   : display name for provenance metadata
    extra_metadata: additional key-value pairs merged into each chunk's metadata

    Returns
    -------
    List of {"text": str, "metadata": dict} dicts
    """
    splitter = _make_splitter()
    lc_docs = splitter.create_documents([text])

    chunks: List[Dict[str, Any]] = []
    for i, doc in enumerate(lc_docs):
        txt = doc.page_content.strip()
        if len(txt) < MIN_CHUNK_LEN:
            continue

        meta: Dict[str, Any] = {
            "contract_id":  contract_id,
            "source":       source_name,
            "chunk_index":  i,
            "start_index":  doc.metadata.get("start_index", 0),
        }
        if extra_metadata:
            meta.update(extra_metadata)

        chunks.append({"text": txt, "metadata": meta})

    logger.debug(
        "Chunked '{}' → {} chunks (contract_id={})",
        source_name, len(chunks), contract_id,
    )
    return chunks


def chunk_pdf(
    pdf_path: str | Path,
    contract_id: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Load a PDF from disk and return semantic clause chunks.

    Parameters
    ----------
    pdf_path    : path to the PDF file
    contract_id : UUID string of the Contract DB row
    extra_metadata: additional key-value pairs merged into each chunk's metadata

    Returns
    -------
    List of {"text": str, "metadata": dict} dicts, same format as chunk_text()
    """
    from langchain_community.document_loaders import PyPDFLoader

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Contract PDF not found: {path}")

    loader = PyPDFLoader(str(path))
    pages  = loader.load()
    logger.info(
        "Loaded PDF '{}' — {} page(s)", path.name, len(pages)
    )

    splitter = _make_splitter()
    lc_docs  = splitter.split_documents(pages)

    chunks: List[Dict[str, Any]] = []
    for i, doc in enumerate(lc_docs):
        txt = doc.page_content.strip()
        if len(txt) < MIN_CHUNK_LEN:
            continue

        meta: Dict[str, Any] = {
            "contract_id":  contract_id,
            "source":       path.name,
            "page":         doc.metadata.get("page", 0) + 1,  # 1-indexed
            "chunk_index":  i,
        }
        if extra_metadata:
            meta.update(extra_metadata)

        chunks.append({"text": txt, "metadata": meta})

    logger.success(
        "PDF '{}' → {} usable chunks for contract_id={}",
        path.name, len(chunks), contract_id,
    )
    return chunks


def chunk_bytes(
    pdf_bytes: bytes,
    contract_id: str,
    filename: str = "upload.pdf",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Load a PDF from in-memory bytes (FastAPI UploadFile.read()) and chunk it.

    Writes the bytes to a temp file, loads, then deletes.
    """
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(
        suffix=".pdf", delete=False, prefix="clm_upload_"
    ) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        return chunk_pdf(
            tmp_path,
            contract_id=contract_id,
            extra_metadata={"original_filename": filename, **(extra_metadata or {})},
        )
    finally:
        os.unlink(tmp_path)
