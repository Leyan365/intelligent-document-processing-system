"""Evaluate semantic search with information retrieval metrics.

This script builds a small in-memory business-document corpus, indexes it
through the existing SemanticSearchService, and evaluates ranked search results
with Precision@K, Recall@K, MRR@K, and NDCG@K.

MRR measures how early the first relevant result appears in the ranked list.
NDCG measures overall ranked result quality by using graded relevance and a
rank discount, so highly relevant documents are rewarded more when they appear
near the top.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from idp_system.pipeline import search as search_module
from idp_system.pipeline.search import SemanticSearchService


DEFAULT_K_VALUES = (1, 3, 5)
DEFAULT_TOP_K = 5
EMBEDDING_DIMENSION = 256
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "or",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True)
class EvaluationQuery:
    """A search query with graded document relevance labels."""

    text: str
    relevant_docs: dict[str, int]


class DeterministicEvaluationEmbeddingService:
    """Small offline embedding service for deterministic search evaluation.

    The production search service defaults to sentence-transformers embeddings.
    The evaluator injects this local lexical embedding service so Phase 12 can
    run without network access, external datasets, or model downloads.
    """

    def __init__(self, reference_texts: list[str], dimension: int = EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension
        self.idf = _build_idf(reference_texts)

    def embed(self, text: str) -> list[float]:
        return self._embed_text(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype="float32")
        tokens = _tokenize(text)
        if not tokens:
            return vector.tolist()

        counts = Counter(tokens)
        for token, count in counts.items():
            index = _stable_token_index(token, self.dimension)
            vector[index] += float(count) * self.idf.get(token, 1.0)

        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector.tolist()


class SimpleInnerProductIndex:
    """Tiny IndexFlatIP-compatible fallback used when faiss-cpu is unavailable."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype="float32")

    def add(self, embeddings: Any) -> None:
        values = np.array(embeddings, dtype="float32")
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("embedding dimension does not match index dimension")
        self.vectors = np.vstack([self.vectors, values])

    def search(self, query_embedding: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
        query = np.array(query_embedding, dtype="float32")
        if query.ndim != 2 or query.shape[0] != 1 or query.shape[1] != self.dimension:
            raise ValueError("query embedding has invalid shape")
        if k <= 0 or len(self.vectors) == 0:
            return (
                np.empty((1, 0), dtype="float32"),
                np.empty((1, 0), dtype="int64"),
            )

        scores = self.vectors @ query[0]
        order = np.argsort(-scores, kind="mergesort")[:k]
        return (
            scores[order].reshape(1, -1).astype("float32"),
            order.reshape(1, -1).astype("int64"),
        )


def build_corpus() -> list[dict[str, object]]:
    """Return a small deterministic corpus of representative business documents."""
    return [
        {
            "id": "invoice_superstore_39519",
            "document_type": "invoice",
            "supplier": "SuperStore",
            "amount": "$22.17",
            "date": "2012-10-23",
            "text": (
                "Invoice document. Supplier SuperStore. Invoice number 39519. "
                "Bill To Aaron Bergman. Invoice date Oct 23 2012. "
                "Balance Due $22.17. Total amount due for office supplies."
            ),
        },
        {
            "id": "invoice_superstore_6817",
            "document_type": "invoice",
            "supplier": "SuperStore",
            "amount": "$10,672.30",
            "date": "2012-10-23",
            "text": (
                "Invoice document. Supplier SuperStore. Invoice number 6817. "
                "Bill To Aaron Hawkins. Invoice date Oct 23 2012. "
                "Balance Due $10,672.30. Grand total for shipped goods."
            ),
        },
        {
            "id": "receipt_quantum_logic_100",
            "document_type": "receipt",
            "supplier": "Quantum Logic Solutions",
            "amount": "Rs. 13,500.00",
            "date": "2026-05-07",
            "text": (
                "Receipt document. Company Name Quantum Logic Solutions. "
                "Receipt number 100. Receipt Date 07/05/2026. "
                "Total AMT PAYABLE Rs. 13,500.00. Paid Amount Rs. 13,500.00. "
                "Customer payment cash."
            ),
        },
        {
            "id": "receipt_aeon_20180314",
            "document_type": "receipt",
            "supplier": "AEON",
            "amount": "GST included total",
            "date": "2018-03-14",
            "text": (
                "Receipt document. AEON tax invoice. TRN CR0005140. "
                "Date 14/02/2018 5:37:44PM. Cashier 2. "
                "Total includes GST. Customer's Payment Cash."
            ),
        },
        {
            "id": "po_screenline_5380034300",
            "document_type": "purchase_order",
            "supplier": "SCREENLINE (PVT) LTD",
            "amount": "5,746.60",
            "date": "2026-01-25",
            "text": (
                "Purchase order document. Supplier SCREENLINE PVT LTD. "
                "PO Number 5380034300. PO Creation Date 25.01.2026. "
                "Grand Total 5,746.60. Buyer requests delivery items."
            ),
        },
        {
            "id": "po_screenline_5380034370",
            "document_type": "purchase_order",
            "supplier": "SCREENLINE (PVT) LTD",
            "amount": "17.52 unit price",
            "date": "2026-01-25",
            "text": (
                "Purchase order document. Supplier SCREENLINE PVT LTD. "
                "PO Number 5380034370. PO Creation Date 25.01.2026. "
                "Item Unit Price 17.52. Delivery address and buyer details."
            ),
        },
    ]


def build_queries() -> list[EvaluationQuery]:
    """Return graded relevance queries covering invoices, receipts, POs, and suppliers."""
    return [
        EvaluationQuery(
            text="SuperStore invoice balance due",
            relevant_docs={
                "invoice_superstore_39519": 3,
                "invoice_superstore_6817": 3,
            },
        ),
        EvaluationQuery(
            text="invoice 39519 Aaron Bergman",
            relevant_docs={
                "invoice_superstore_39519": 3,
                "invoice_superstore_6817": 1,
            },
        ),
        EvaluationQuery(
            text="invoice 6817 Aaron Hawkins",
            relevant_docs={
                "invoice_superstore_6817": 3,
                "invoice_superstore_39519": 1,
            },
        ),
        EvaluationQuery(
            text="receipt Quantum Logic Solutions total 13500",
            relevant_docs={
                "receipt_quantum_logic_100": 3,
            },
        ),
        EvaluationQuery(
            text="AEON receipt paid by cash GST",
            relevant_docs={
                "receipt_aeon_20180314": 3,
                "receipt_quantum_logic_100": 1,
            },
        ),
        EvaluationQuery(
            text="purchase order Screenline supplier",
            relevant_docs={
                "po_screenline_5380034300": 3,
                "po_screenline_5380034370": 3,
            },
        ),
        EvaluationQuery(
            text="PO number 5380034300",
            relevant_docs={
                "po_screenline_5380034300": 3,
                "po_screenline_5380034370": 1,
            },
        ),
        EvaluationQuery(
            text="documents from Screenline",
            relevant_docs={
                "po_screenline_5380034300": 3,
                "po_screenline_5380034370": 3,
            },
        ),
        EvaluationQuery(
            text="receipt paid by cash",
            relevant_docs={
                "receipt_aeon_20180314": 3,
                "receipt_quantum_logic_100": 2,
            },
        ),
        EvaluationQuery(
            text="business document with total amount",
            relevant_docs={
                "invoice_superstore_39519": 2,
                "invoice_superstore_6817": 2,
                "receipt_quantum_logic_100": 2,
                "receipt_aeon_20180314": 1,
                "po_screenline_5380034300": 1,
                "po_screenline_5380034370": 1,
            },
        ),
    ]


def precision_at_k(results: list[dict[str, object]], relevant_docs: dict[str, int], k: int) -> float:
    """Return relevant retrieved in top K divided by K."""
    if k <= 0:
        return 0.0
    relevant_ids = _binary_relevant_ids(relevant_docs)
    retrieved_ids = _result_ids(results)[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant_ids)
    return relevant_retrieved / k


def recall_at_k(results: list[dict[str, object]], relevant_docs: dict[str, int], k: int) -> float:
    """Return relevant retrieved in top K divided by all relevant documents."""
    relevant_ids = _binary_relevant_ids(relevant_docs)
    if not relevant_ids or k <= 0:
        return 0.0
    retrieved_ids = _result_ids(results)[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant_ids)
    return relevant_retrieved / len(relevant_ids)


def reciprocal_rank_at_k(results: list[dict[str, object]], relevant_docs: dict[str, int], k: int) -> float:
    """Return 1/rank for the first relevant result within K, otherwise 0."""
    relevant_ids = _binary_relevant_ids(relevant_docs)
    if not relevant_ids or k <= 0:
        return 0.0
    for rank, doc_id in enumerate(_result_ids(results)[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(results: list[dict[str, object]], relevant_docs: dict[str, int], k: int) -> float:
    """Return discounted cumulative gain at K using graded relevance."""
    if k <= 0:
        return 0.0
    dcg = 0.0
    for rank, doc_id in enumerate(_result_ids(results)[:k], start=1):
        relevance = max(0, relevant_docs.get(doc_id, 0))
        gain = (2**relevance) - 1
        discount = math.log2(rank + 1)
        dcg += gain / discount
    return dcg


def ndcg_at_k(results: list[dict[str, object]], relevant_docs: dict[str, int], k: int) -> float:
    """Return normalized DCG at K against the ideal graded ranking."""
    if k <= 0:
        return 0.0
    ideal_relevances = sorted(
        (max(0, relevance) for relevance in relevant_docs.values()),
        reverse=True,
    )[:k]
    ideal_dcg = 0.0
    for rank, relevance in enumerate(ideal_relevances, start=1):
        ideal_dcg += ((2**relevance) - 1) / math.log2(rank + 1)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(results, relevant_docs, k) / ideal_dcg


def evaluate_queries(
    service: SemanticSearchService,
    queries: list[EvaluationQuery],
    k_values: tuple[int, ...],
    top_k: int,
) -> list[dict[str, object]]:
    """Run all queries and collect per-query metrics."""
    evaluations: list[dict[str, object]] = []
    for query in queries:
        results = service.search(query.text, k=top_k)
        metrics = {
            k: {
                "precision": precision_at_k(results, query.relevant_docs, k),
                "recall": recall_at_k(results, query.relevant_docs, k),
                "rr": reciprocal_rank_at_k(results, query.relevant_docs, k),
                "ndcg": ndcg_at_k(results, query.relevant_docs, k),
            }
            for k in k_values
        }
        evaluations.append(
            {
                "query": query,
                "results": results,
                "metrics": metrics,
            }
        )
    return evaluations


def aggregate_metrics(
    evaluations: list[dict[str, object]],
    k_values: tuple[int, ...],
) -> dict[int, dict[str, float]]:
    """Compute mean Precision, Recall, MRR, and NDCG for each K."""
    aggregates: dict[int, dict[str, float]] = {}
    query_count = len(evaluations)
    for k in k_values:
        if query_count == 0:
            aggregates[k] = {
                "mean_precision": 0.0,
                "mean_recall": 0.0,
                "mrr": 0.0,
                "mean_ndcg": 0.0,
            }
            continue

        metrics_by_query = [
            evaluation["metrics"][k]  # type: ignore[index]
            for evaluation in evaluations
        ]
        aggregates[k] = {
            "mean_precision": _mean(metric["precision"] for metric in metrics_by_query),
            "mean_recall": _mean(metric["recall"] for metric in metrics_by_query),
            "mrr": _mean(metric["rr"] for metric in metrics_by_query),
            "mean_ndcg": _mean(metric["ndcg"] for metric in metrics_by_query),
        }
    return aggregates


def find_weak_queries(
    evaluations: list[dict[str, object]],
    max_k: int,
    min_ndcg: float = 0.85,
    min_recall: float = 0.75,
) -> list[dict[str, object]]:
    """Return queries whose largest-K ranking suggests weak retrieval."""
    weak: list[dict[str, object]] = []
    for evaluation in evaluations:
        metrics = evaluation["metrics"][max_k]  # type: ignore[index]
        if metrics["rr"] == 0.0 or metrics["ndcg"] < min_ndcg or metrics["recall"] < min_recall:
            weak.append(evaluation)
    return weak


def print_report(
    corpus: list[dict[str, object]],
    queries: list[EvaluationQuery],
    evaluations: list[dict[str, object]],
    aggregates: dict[int, dict[str, float]],
    k_values: tuple[int, ...],
    top_k: int,
    backend_label: str,
) -> None:
    """Print a human-readable evaluation report."""
    print("Semantic Search Evaluation")
    print(f"- corpus_size: {len(corpus)}")
    print(f"- query_count: {len(queries)}")
    print(f"- k_values: {', '.join(str(k) for k in k_values)}")
    print(f"- top_k: {top_k}")
    print(f"- embedding_backend: deterministic offline lexical embeddings")
    print(f"- search_backend: {backend_label}")
    print()

    for evaluation in evaluations:
        query = evaluation["query"]
        results = evaluation["results"]
        metrics = evaluation["metrics"]

        print(f"Query: {query.text}")
        print(f"  Top returned document IDs: {', '.join(_result_ids(results)) or '<none>'}")
        print(f"  Relevant document IDs: {_format_relevant_docs(query.relevant_docs)}")
        for k in k_values:
            metric = metrics[k]
            print(
                f"  K={k}: "
                f"Precision={metric['precision']:.4f}, "
                f"Recall={metric['recall']:.4f}, "
                f"RR={metric['rr']:.4f}, "
                f"NDCG={metric['ndcg']:.4f}"
            )
        print()

    print("Aggregate Metrics")
    for k in k_values:
        metric = aggregates[k]
        print(f"- Mean Precision@{k}: {metric['mean_precision']:.4f}")
        print(f"- Mean Recall@{k}: {metric['mean_recall']:.4f}")
        print(f"- MRR@{k}: {metric['mrr']:.4f}")
        print(f"- Mean NDCG@{k}: {metric['mean_ndcg']:.4f}")
    print()

    weak_queries = find_weak_queries(evaluations, max(k_values))
    print("Weak Performing Queries")
    if not weak_queries:
        print("- none")
        return
    for evaluation in weak_queries:
        query = evaluation["query"]
        metrics = evaluation["metrics"][max(k_values)]
        print(
            f"- {query.text}: "
            f"Recall@{max(k_values)}={metrics['recall']:.4f}, "
            f"RR@{max(k_values)}={metrics['rr']:.4f}, "
            f"NDCG@{max(k_values)}={metrics['ndcg']:.4f}"
        )


def build_search_service(
    corpus: list[dict[str, object]],
    queries: list[EvaluationQuery],
) -> tuple[SemanticSearchService, str]:
    """Build the existing SemanticSearchService with offline evaluation embeddings."""
    backend_label = configure_search_backend()
    reference_texts = [str(document["text"]) for document in corpus]
    reference_texts.extend(query.text for query in queries)
    service = SemanticSearchService(
        embedding_service=DeterministicEvaluationEmbeddingService(reference_texts)
    )
    service.add_documents(corpus)
    return service, backend_label


def configure_search_backend() -> str:
    """Use FAISS when installed, otherwise patch in a local compatible index."""
    if importlib.util.find_spec("faiss") is not None:
        return "faiss IndexFlatIP"

    search_module._create_faiss_index = SimpleInnerProductIndex  # type: ignore[attr-defined]
    return "in-script IndexFlatIP-compatible fallback"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the local semantic search service with IR metrics."
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_VALUES),
        help="K values for Precision, Recall, MRR, and NDCG. Default: 1 3 5.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of search results to retrieve per query. Default: 5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = tuple(sorted({k for k in args.k_values if k > 0}))
    if not k_values:
        raise ValueError("--k-values must include at least one positive integer")

    top_k = max(1, args.top_k)
    if top_k < max(k_values):
        top_k = max(k_values)

    corpus = build_corpus()
    queries = build_queries()
    service, backend_label = build_search_service(corpus, queries)
    evaluations = evaluate_queries(service, queries, k_values, top_k)
    aggregates = aggregate_metrics(evaluations, k_values)
    print_report(corpus, queries, evaluations, aggregates, k_values, top_k, backend_label)


def _tokenize(text: str) -> list[str]:
    tokens = [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if token not in STOPWORDS
    ]
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(_synonyms_for(token))
    return expanded


def _synonyms_for(token: str) -> tuple[str, ...]:
    synonyms = {
        "po": ("purchase", "order"),
        "purchase": ("po",),
        "order": ("po",),
        "balance": ("total", "amount"),
        "due": ("payable",),
        "payable": ("due", "amount"),
        "paid": ("payment", "cash"),
        "payment": ("paid",),
        "supplier": ("vendor", "company"),
        "company": ("supplier",),
        "screenline": ("supplier", "vendor"),
        "superstore": ("supplier", "vendor"),
        "aeon": ("supplier", "receipt"),
        "quantum": ("supplier", "receipt"),
        "logic": ("supplier", "receipt"),
        "gst": ("tax",),
        "tax": ("gst",),
    }
    return synonyms.get(token, ())


def _build_idf(texts: list[str]) -> dict[str, float]:
    document_count = len(texts)
    document_frequency: Counter[str] = Counter()
    for text in texts:
        document_frequency.update(set(_tokenize(text)))

    return {
        token: math.log((1 + document_count) / (1 + frequency)) + 1.0
        for token, frequency in document_frequency.items()
    }


def _stable_token_index(token: str, dimension: int) -> int:
    value = 0
    for char in token:
        value = ((value * 131) + ord(char)) % dimension
    return value


def _binary_relevant_ids(relevant_docs: dict[str, int]) -> set[str]:
    return {doc_id for doc_id, relevance in relevant_docs.items() if relevance > 0}


def _result_ids(results: list[dict[str, object]]) -> list[str]:
    return [str(result.get("id", "")) for result in results if result.get("id")]


def _format_relevant_docs(relevant_docs: dict[str, int]) -> str:
    if not relevant_docs:
        return "<none>"
    return ", ".join(f"{doc_id}:{grade}" for doc_id, grade in relevant_docs.items())


def _mean(values: Any) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(float(value) for value in materialized) / len(materialized)


if __name__ == "__main__":
    main()
