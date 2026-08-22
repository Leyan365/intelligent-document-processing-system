"""Regression tests for the production-shaped search evaluation harness."""

from __future__ import annotations

import unittest

from evaluation.search_eval import (
    build_corpus,
    build_queries,
    build_search_service,
    evaluate_queries,
    validate_corpus_schema,
)


class SearchEvaluationHarnessTests(unittest.TestCase):
    def test_corpus_uses_production_search_shape(self) -> None:
        corpus = build_corpus()
        validate_corpus_schema(corpus)
        for document in corpus:
            self.assertIn("type", document)
            self.assertIn("fields", document)
            self.assertNotIn("document_type", document)

    def test_malformed_corpus_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing top-level fields"):
            validate_corpus_schema([{"id": "broken", "text": "text"}])

    def test_deterministic_harness_only_misses_expected_no_match(self) -> None:
        corpus = build_corpus()
        queries = build_queries()
        validate_corpus_schema(corpus)
        service, embedding_label, _ = build_search_service(
            corpus,
            queries,
            "deterministic",
        )
        evaluations = evaluate_queries(service, queries, (1, 3, 5), 5)
        empty_queries = [evaluation["query"] for evaluation in evaluations if not evaluation["results"]]

        self.assertIn("fallback/test only", embedding_label)
        self.assertEqual(1, len(empty_queries))
        self.assertTrue(empty_queries[0].expects_no_results)


if __name__ == "__main__":
    unittest.main()
