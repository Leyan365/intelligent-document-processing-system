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

    def search(self, query: str, k: int = 5, min_score: float = 0.0) -> list[dict[str, object]]:
        """Return top-k semantically similar in-memory documents."""
        if not query or not query.strip():
            raise ValueError("query is required")
        if self.index is None or not self.documents:
            return []

        from .query_parser import parse_query, parse_amount

        known_suppliers = []
        for doc in self.documents:
            fields = doc.get("fields") or {}
            sup = fields.get("supplier")
            if sup and isinstance(sup, str):
                known_suppliers.append(sup)

        parsed_query = parse_query(query, known_suppliers=list(set(known_suppliers)))

        has_structured_constraints = (
            parsed_query.document_type is not None or
            parsed_query.supplier is not None or
            parsed_query.document_number is not None or
            parsed_query.amount_eq is not None or
            parsed_query.amount_lt is not None or
            parsed_query.amount_lte is not None or
            parsed_query.amount_gt is not None or
            parsed_query.amount_gte is not None or
            parsed_query.amount_min is not None or
            parsed_query.amount_max is not None
        )

        candidate_indices = []
        for i, doc in enumerate(self.documents):
            passed = True
            if parsed_query.document_type and doc.get("type") != parsed_query.document_type:
                passed = False

            if passed and parsed_query.supplier:
                from .query_parser import normalize_supplier_name
                fields = doc.get("fields") or {}
                doc_supplier = str(fields.get("supplier", ""))

                norm_query = normalize_supplier_name(parsed_query.supplier)
                norm_doc = normalize_supplier_name(doc_supplier)

                if norm_query != norm_doc:
                    passed = False

            if passed and parsed_query.document_number:
                fields = doc.get("fields") or {}
                doc_inv = str(fields.get("invoice_number", "")).lower()
                doc_num_query = parsed_query.document_number.lower()
                if doc_num_query not in doc_inv:
                    passed = False

            if passed and (parsed_query.amount_eq is not None or
                           parsed_query.amount_lt is not None or
                           parsed_query.amount_lte is not None or
                           parsed_query.amount_gt is not None or
                           parsed_query.amount_gte is not None or
                           parsed_query.amount_min is not None):
                fields = doc.get("fields") or {}
                doc_amt_str = str(fields.get("amount", ""))
                doc_amt = parse_amount(doc_amt_str)
                if doc_amt is None:
                    passed = False
                else:
                    if parsed_query.amount_eq is not None and doc_amt != parsed_query.amount_eq:
                        passed = False
                    if parsed_query.amount_lt is not None and doc_amt >= parsed_query.amount_lt:
                        passed = False
                    if parsed_query.amount_lte is not None and doc_amt > parsed_query.amount_lte:
                        passed = False
                    if parsed_query.amount_gt is not None and doc_amt <= parsed_query.amount_gt:
                        passed = False
                    if parsed_query.amount_gte is not None and doc_amt < parsed_query.amount_gte:
                        passed = False
                    if parsed_query.amount_min is not None and (doc_amt < parsed_query.amount_min or doc_amt > parsed_query.amount_max):
                        passed = False

            if passed:
                candidate_indices.append(i)

        if has_structured_constraints and not candidate_indices:
            return []

        if not candidate_indices:
            candidate_indices = list(range(len(self.documents)))

        semantic_text = parsed_query.semantic_text
        if not semantic_text:
            results = []
            for idx in candidate_indices[:k]:
                doc = self.documents[idx]
                results.append(
                    {
                        "id": str(doc.get("id", idx)),
                        "text": str(doc.get("text", "")),
                        "score": 1.0,
                        "metadata": {
                            key: value
                            for key, value in doc.items()
                            if key not in {"id", "text"}
                        },
                    }
                )
            return results

        query_embedding = np.array([self.embedding_service.embed(semantic_text)], dtype="float32")
        scores, indices = self.index.search(query_embedding, len(self.documents))

        candidate_set = set(candidate_indices)
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index not in candidate_set:
                continue

            if not has_structured_constraints and score < min_score:
                continue

            doc = self.documents[int(index)]
            results.append(
                {
                    "id": str(doc.get("id", index)),
                    "text": str(doc.get("text", "")),
                    "score": float(score),
                    "metadata": {
                        key: value
                        for key, value in doc.items()
                        if key not in {"id", "text"}
                    },
                }
            )
            if len(results) >= k:
                break

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
