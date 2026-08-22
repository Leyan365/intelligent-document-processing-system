"""Regression tests for semantic ranking and optional score thresholds."""

from __future__ import annotations

import unittest

from idp_system.pipeline.search import SemanticSearchService


class NegativeScoreEmbeddingService:
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "first unrelated wording": [-0.4, 0.0],
            "second unrelated wording": [-0.2, 0.0],
            "receipt candidate": [0.2, 0.0],
            "invoice candidate": [0.9, 0.0],
        }
        return [vectors[text] for text in texts]

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


class SemanticThresholdTests(unittest.TestCase):
    def test_default_semantic_search_keeps_top_candidates_with_negative_scores(self) -> None:
        service = SemanticSearchService(embedding_service=NegativeScoreEmbeddingService())
        service.add_documents(
            [
                {"id": "first", "text": "first unrelated wording"},
                {"id": "second", "text": "second unrelated wording"},
            ]
        )

        results = service.search("conceptual intent", k=2)

        self.assertEqual(["second", "first"], [result["id"] for result in results])

    def test_explicit_semantic_threshold_is_still_enforced(self) -> None:
        service = SemanticSearchService(embedding_service=NegativeScoreEmbeddingService())
        service.add_documents(
            [
                {"id": "first", "text": "first unrelated wording"},
                {"id": "second", "text": "second unrelated wording"},
            ]
        )

        results = service.search("conceptual intent", k=2, min_score=-0.3)

        self.assertEqual(["second"], [result["id"] for result in results])

    def test_structured_filter_runs_before_semantic_ranking(self) -> None:
        service = SemanticSearchService(embedding_service=NegativeScoreEmbeddingService())
        service.add_documents(
            [
                {
                    "id": "receipt",
                    "text": "receipt candidate",
                    "type": "receipt",
                    "fields": {},
                },
                {
                    "id": "invoice",
                    "text": "invoice candidate",
                    "type": "invoice",
                    "fields": {},
                },
            ]
        )

        results = service.search("receipt customer intent", k=2)

        self.assertEqual(["receipt"], [result["id"] for result in results])


if __name__ == "__main__":
    unittest.main()
