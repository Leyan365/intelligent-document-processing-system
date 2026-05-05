"""In-memory semantic search with sentence-transformers and FAISS."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .embeddings import EmbeddingService


EXAMPLE_DOCUMENTS = [
    {
        "id": "doc-1",
        "text": "Invoice INV-1001 from Acme Office Supplies with total amount due.",
    },
    {
        "id": "doc-2",
        "text": "Receipt for grocery purchase paid by card at the cashier.",
    },
    {
        "id": "doc-3",
        "text": "Purchase order PO-450 requesting laptop accessories from a vendor.",
    },
]


@dataclass(slots=True)
class SearchResult:
    """A single semantic search hit."""

    id: str
    text: str
    score: float
    metadata: dict[str, object] = field(default_factory=dict)


class SemanticSearchService:
    """Store document embeddings in memory and query them with FAISS."""

    def __init__(self, embedding_service: EmbeddingService | None = None) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.documents: list[dict[str, object]] = []
        self.index: Any | None = None
        self.dimension: int | None = None

    def add_documents(self, documents: list[dict[str, object]]) -> None:
        """Embed and add documents shaped as {'id': ..., 'text': ..., ...}."""
        if not documents:
            return

        texts = [str(document.get("text", "")) for document in documents]
        embeddings = np.array(self.embedding_service.embed_many(texts), dtype="float32")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(documents):
            raise ValueError("embedding service returned invalid embedding shape")

        if self.index is None:
            self.dimension = int(embeddings.shape[1])
            self.index = _create_faiss_index(self.dimension)
        elif embeddings.shape[1] != self.dimension:
            raise ValueError("embedding dimension does not match existing index")

        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query: str, k: int = 5) -> list[dict[str, object]]:
        """Return top-k semantically similar in-memory documents."""
        if not query or not query.strip():
            raise ValueError("query is required")
        if self.index is None or not self.documents:
            return []

        query_embedding = np.array([self.embedding_service.embed(query)], dtype="float32")
        scores, indices = self.index.search(query_embedding, min(k, len(self.documents)))

        results: list[dict[str, object]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            document = self.documents[int(index)]
            results.append(
                {
                    "id": str(document.get("id", index)),
                    "text": str(document.get("text", "")),
                    "score": float(score),
                    "metadata": {
                        key: value
                        for key, value in document.items()
                        if key not in {"id", "text"}
                    },
                }
            )
        return results


def build_example_search_service() -> SemanticSearchService:
    """Build a tiny in-memory search index for smoke testing."""
    service = SemanticSearchService()
    service.add_documents(EXAMPLE_DOCUMENTS)
    return service


def _create_faiss_index(dimension: int) -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is required for semantic search. Install package: faiss-cpu"
        ) from exc

    return faiss.IndexFlatIP(dimension)


if __name__ == "__main__":
    search_service = build_example_search_service()
    print(search_service.search("find the office supplies invoice", k=2))
