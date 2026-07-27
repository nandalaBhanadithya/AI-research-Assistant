from functools import lru_cache
from typing import Optional

import chromadb

from app.config import get_settings

COLLECTION_NAME = "chunks"


@lru_cache
def _client() -> chromadb.ClientAPI:
    settings = get_settings()
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


@lru_cache
def _collection():
    return _client().get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def add_chunks(
    chunk_ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]
) -> None:
    if not chunk_ids:
        return
    _collection().add(ids=chunk_ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def delete_document(document_id: str) -> None:
    _collection().delete(where={"document_id": document_id})


def query(
    embedding: list[float], top_k: int, document_ids: Optional[list[str]] = None
) -> list[dict]:
    where = {"document_id": {"$in": document_ids}} if document_ids else None
    result = _collection().query(
        query_embeddings=[embedding], n_results=top_k, where=where, include=["documents", "metadatas", "distances"]
    )
    if not result["ids"] or not result["ids"][0]:
        return []

    hits = []
    for chunk_id, doc_text, metadata, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        similarity = max(0.0, 1.0 - distance)
        hits.append({"chunk_id": chunk_id, "text": doc_text, "metadata": metadata, "similarity": similarity})
    return hits


def count() -> int:
    return _collection().count()
