"""In-memory semantic search with sentence-transformers and FAISS."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .embeddings import EmbeddingModelUnavailableError, EmbeddingService


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
        self.semantic_search_available = True
        self.semantic_search_error: str | None = None

    def add_documents(self, documents: list[dict[str, object]]) -> None:
        """Embed and add documents shaped as {'id': ..., 'text': ..., ...}."""
        if not documents:
            return

        texts = [str(document.get("text", "")) for document in documents]
        try:
            embeddings = np.array(self.embedding_service.embed_many(texts), dtype="float32")
        except EmbeddingModelUnavailableError as exc:
            self.documents.extend(documents)
            self.index = None
            self.dimension = None
            self.semantic_search_available = False
            self.semantic_search_error = str(exc)
            return

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
        if not self.documents:
            return []

        from .query_parser import parse_date, parse_query, parse_amount

        known_suppliers = []
        for doc in self.documents:
            fields = doc.get("fields") or {}
            sup = fields.get("supplier")
            if sup and isinstance(sup, str):
                known_suppliers.append(sup)

        parsed_query = parse_query(query, known_suppliers=list(set(known_suppliers)))
        if parsed_query.date_error:
            raise ValueError(parsed_query.date_error)

        has_amount_constraint = (
            parsed_query.amount_eq is not None or
            parsed_query.amount_lt is not None or
            parsed_query.amount_lte is not None or
            parsed_query.amount_gt is not None or
            parsed_query.amount_gte is not None or
            parsed_query.amount_min is not None or
            parsed_query.amount_max is not None
        )
        has_date_constraint = (
            parsed_query.date_eq is not None or
            parsed_query.date_lt is not None or
            parsed_query.date_lte is not None or
            parsed_query.date_gt is not None or
            parsed_query.date_gte is not None or
            parsed_query.date_min is not None or
            parsed_query.date_max is not None
        )

        has_structured_constraints = (
            parsed_query.document_type is not None or
            parsed_query.supplier is not None or
            parsed_query.document_number is not None or
            has_amount_constraint or
            has_date_constraint
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

            if passed and has_amount_constraint:
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

            if passed and has_date_constraint:
                fields = doc.get("fields") or {}
                doc_date = parse_date(str(fields.get("date", "")))
                if doc_date is None:
                    passed = False
                else:
                    if parsed_query.date_eq is not None and doc_date != parsed_query.date_eq:
                        passed = False
                    if parsed_query.date_lt is not None and doc_date >= parsed_query.date_lt:
                        passed = False
                    if parsed_query.date_lte is not None and doc_date > parsed_query.date_lte:
                        passed = False
                    if parsed_query.date_gt is not None and doc_date <= parsed_query.date_gt:
                        passed = False
                    if parsed_query.date_gte is not None and doc_date < parsed_query.date_gte:
                        passed = False
                    if parsed_query.date_min is not None and (
                        doc_date < parsed_query.date_min or doc_date > parsed_query.date_max
                    ):
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

        def _compute_lexical_tier(doc: dict, query_text: str) -> int:
            from .query_parser import normalize_supplier_name
            import re
            q_norm = normalize_supplier_name(query_text)
            if not q_norm:
                return 5

            fields = doc.get("fields") or {}

            # Tier 1: exact document number
            doc_num = normalize_supplier_name(str(fields.get("invoice_number", "")))
            if doc_num and doc_num == q_norm:
                return 1

            # Tier 2: exact supplier phrase
            doc_sup = normalize_supplier_name(str(fields.get("supplier", "")))
            pattern = r'\b' + re.escape(q_norm) + r'\b'
            if doc_sup and re.search(pattern, doc_sup):
                return 2

            # Tier 3: filename
            filename = normalize_supplier_name(str(doc.get("filename") or ""))
            if filename and re.search(pattern, filename):
                return 3

            # Tier 4: text
            text = normalize_supplier_name(str(doc.get("text") or ""))
            if text and re.search(pattern, text):
                return 4

            return 5

        # Exact textual matches take precedence over dense similarity. Semantic
        # fallback is only for queries with no Tier 1-4 match anywhere in the
        # corpus, so an unrelated document cannot be appended to an entity
        # search merely to fill the requested result count.
        requires_exact_lexical_match = (
            bool(semantic_text)
            and any(_compute_lexical_tier(doc, semantic_text) < 5 for doc in self.documents)
        )
        if requires_exact_lexical_match and not any(
            _compute_lexical_tier(self.documents[index], semantic_text) < 5
            for index in candidate_indices
        ):
            return []

        if self.index is None:
            return _lexical_search_results(
                self.documents,
                candidate_indices,
                semantic_text,
                k,
                _compute_lexical_tier,
                requires_exact_lexical_match,
            )

        query_embedding = np.array([self.embedding_service.embed(semantic_text)], dtype="float32")
        scores, indices = self.index.search(query_embedding, len(self.documents))

        candidate_set = set(candidate_indices)

        scored_candidates = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index not in candidate_set:
                continue

            doc = self.documents[int(index)]
            tier = _compute_lexical_tier(doc, semantic_text)

            if requires_exact_lexical_match and tier == 5:
                continue

            if tier == 5 and not has_structured_constraints and score < min_score:
                continue

            scored_candidates.append((tier, float(score), doc, int(index)))

        scored_candidates.sort(key=lambda x: (x[0], -x[1]))

        results = []
        for tier, score, doc, index in scored_candidates[:k]:
            results.append(
                {
                    "id": str(doc.get("id", index)),
                    "text": str(doc.get("text", "")),
                    "score": score,
                    "lexical_tier": tier,
                    "metadata": {
                        key: value
                        for key, value in doc.items()
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


def _lexical_search_results(
    documents: list[dict[str, object]],
    candidate_indices: list[int],
    query_text: str,
    k: int,
    compute_lexical_tier: Any,
    requires_exact_lexical_match: bool,
) -> list[dict[str, object]]:
    """Provide fast local search while the optional embedding model is unavailable."""
    from .query_parser import normalize_supplier_name

    query_terms = set(normalize_supplier_name(query_text).split())
    ranked: list[tuple[int, float, dict[str, object], int]] = []
    for index in candidate_indices:
        doc = documents[index]
        tier = compute_lexical_tier(doc, query_text)
        if requires_exact_lexical_match and tier == 5:
            continue

        fields = doc.get("fields") or {}
        searchable_text = " ".join(
            (
                str(doc.get("filename") or ""),
                str(fields.get("supplier") or ""),
                str(doc.get("text") or ""),
            )
        )
        terms = set(normalize_supplier_name(searchable_text).split())
        score = len(query_terms & terms) / len(query_terms) if query_terms else 0.0
        ranked.append((tier, score, doc, index))

    ranked.sort(key=lambda item: (item[0], -item[1]))
    return [
        {
            "id": str(doc.get("id", index)),
            "text": str(doc.get("text", "")),
            "score": score,
            "lexical_tier": tier,
            "metadata": {
                key: value
                for key, value in doc.items()
                if key not in {"id", "text"}
            },
        }
        for tier, score, doc, index in ranked[:k]
    ]


if __name__ == "__main__":
    search_service = build_example_search_service()
    print(search_service.search("find the office supplies invoice", k=2))
