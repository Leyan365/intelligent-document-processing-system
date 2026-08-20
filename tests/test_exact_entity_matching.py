import unittest
from idp_system.pipeline.embeddings import EmbeddingModelUnavailableError
from idp_system.pipeline.search import SemanticSearchService

class MockEmbeddingService:
    def embed_many(self, texts):
        return [[0.1, 0.2]] * len(texts)
    def embed(self, text):
        return [0.1, 0.2]


class UnavailableEmbeddingService:
    def embed_many(self, texts):
        raise EmbeddingModelUnavailableError("model is not cached locally")

    def embed(self, text):
        raise EmbeddingModelUnavailableError("model is not cached locally")

class TestExactEntityMatching(unittest.TestCase):
    def test_entity_matching(self):
        service = SemanticSearchService(embedding_service=MockEmbeddingService())
        # To simulate FAISS bug, we can't easily mock inner FAISS similarities exactly without monkeypatching
        # But we can just test if the sort order correctly puts the Tier matches first if we mock the scores identically
        # Actually, if we mock the scores identically, they will be tied on score, but our sort uses `(tier, -score)`
        # So tier will strictly order them.
        docs = [
            {"id": "docA", "filename": "Ansell_PO_001.pdf", "text": "... ANSELL ... purchase order ...", "type": "purchase_order", "fields": {"amount": "3800"}},
            {"id": "docB", "filename": "Midas_PO_002.pdf", "text": "... MIDAS ... purchase order ...", "type": "purchase_order", "fields": {"amount": "1500"}},
            {"id": "docC", "filename": "Superstore_PO_003.pdf", "text": "... SUPERSTORE ...", "type": "invoice", "fields": {"amount": "2200", "supplier": "Superstore"}},
            {"id": "docD", "filename": "Lalan_PO_004.pdf", "text": "... LALAN ...", "type": "purchase_order", "fields": {}},
            {"id": "docE", "filename": "Ansell_PO_005.pdf", "text": "... ANSELL ...", "type": "purchase_order", "fields": {"amount": "4000"}},
            {"id": "docF", "filename": "Midas_PO_006.pdf", "text": "... MIDAS ...", "type": "purchase_order", "fields": {"amount": "5000"}},
        ]
        service.add_documents(docs)

        res = service.search("Ansell")
        self.assertIn(res[0]["id"], {"docA", "docE"})

        res = service.search("Midas")
        self.assertIn(res[0]["id"], {"docB", "docF"})

        res = service.search("Lalan")
        self.assertEqual(res[0]["id"], "docD")

        res = service.search("Superstore")
        self.assertEqual(res[0]["id"], "docC")

        res = service.search("Ansell below 4000")
        self.assertEqual([result["id"] for result in res], ["docA"])
        self.assertTrue(all(result["lexical_tier"] < 5 for result in res))

        res = service.search("Midas above 3000")
        self.assertEqual([result["id"] for result in res], ["docF"])
        self.assertTrue(all(result["lexical_tier"] < 5 for result in res))

        res = service.search("office equipment")
        self.assertTrue(len(res) > 0)

    def test_entity_and_amount_returns_no_unrelated_fallback(self):
        service = SemanticSearchService(embedding_service=MockEmbeddingService())
        service.add_documents([
            {"id": "ansell", "filename": "Ansell_PO.pdf", "text": "... ANSELL ...", "fields": {"amount": "3800"}},
            {"id": "midas", "filename": "Midas_PO.pdf", "text": "... MIDAS ...", "fields": {"amount": "1500"}},
        ])

        self.assertEqual(service.search("Midas above 3000"), [])

    def test_offline_embedding_fallback_keeps_exact_amount_search(self):
        service = SemanticSearchService(embedding_service=UnavailableEmbeddingService())
        service.add_documents([
            {"id": "ansell", "filename": "Ansell_PO.pdf", "text": "Safety gloves from Ansell", "fields": {"amount": "3800"}},
            {"id": "midas", "filename": "Midas_PO.pdf", "text": "Safety gloves from Midas", "fields": {"amount": "1500"}},
        ])

        self.assertFalse(service.semantic_search_available)
        self.assertEqual([result["id"] for result in service.search("Ansell below 4000")], ["ansell"])
        self.assertEqual(service.search("Midas above 3000"), [])

if __name__ == '__main__':
    unittest.main()
