import unittest
from idp_system.pipeline.search import SemanticSearchService
from decimal import Decimal

class MockEmbeddingService:
    def embed_many(self, texts):
        return [[0.1, 0.2]] * len(texts)
    def embed(self, text):
        return [0.1, 0.2]

class TestHybridSearch(unittest.TestCase):
    def test_hybrid_search(self):
        service = SemanticSearchService(embedding_service=MockEmbeddingService())
        docs = [
            {"id": "doc1", "text": "PO 1", "type": "purchase_order", "fields": {"amount": "3800.00", "supplier": "Vendor A", "date": "25.01.2026"}},
            {"id": "doc2", "text": "PO 2", "type": "purchase_order", "fields": {"amount": "1500.00", "supplier": "Vendor B", "date": "2026-01-30"}},
            {"id": "doc3", "text": "INV 1", "type": "invoice", "fields": {"amount": "5000", "supplier": "Vendor C", "date": "25-Nov-2025"}},
        ]
        service.add_documents(docs)

        # Exact type + below 3000
        res = service.search("purchase orders below 3000")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], "doc2")

        # Type invoice + over 4000
        res2 = service.search("invoices above 4000")
        self.assertEqual(len(res2), 1)
        self.assertEqual(res2[0]["id"], "doc3")

        # Between
        res3 = service.search("between 1000 and 3000")
        self.assertEqual(len(res3), 1)
        self.assertEqual(res3[0]["id"], "doc2")

        # No match
        res4 = service.search("below 1000")
        self.assertEqual(len(res4), 0)

        # Pure semantic search
        res5 = service.search("some arbitrary text")
        self.assertEqual(len(res5), 3)

        # Natural-language query dates work with stored day-first, ISO, and text dates.
        res6 = service.search("purchase orders after 1st January 2026")
        self.assertEqual({result["id"] for result in res6}, {"doc1", "doc2"})

        res7 = service.search("between 1.1.2026 and 31.1.2026")
        self.assertEqual({result["id"] for result in res7}, {"doc1", "doc2"})

        res8 = service.search("before 1st January 2026")
        self.assertEqual([result["id"] for result in res8], ["doc3"])

        # Date aliases and amount filters are applied together.
        res9 = service.search("below 1st February 2026 amount above 300")
        self.assertEqual({result["id"] for result in res9}, {"doc1", "doc2", "doc3"})

        with self.assertRaisesRegex(ValueError, "complete day, month, and year"):
            service.search("after 2026")

if __name__ == '__main__':
    unittest.main()
