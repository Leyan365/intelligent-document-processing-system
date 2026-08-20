import unittest
from idp_system.pipeline.search import SemanticSearchService
from idp_system.pipeline.query_parser import parse_query

class MockEmbeddingService:
    def embed_many(self, texts):
        return [[0.1, 0.2]] * len(texts)
    def embed(self, text):
        return [0.1, 0.2]

class TestSupplierMatching(unittest.TestCase):
    def test_supplier_matching(self):
        service = SemanticSearchService(embedding_service=MockEmbeddingService())
        docs = [
            {"id": "doc1", "text": "PO 1", "type": "purchase_order", "fields": {"amount": "3800.00", "supplier": "Ansell"}},
            {"id": "doc2", "text": "PO 2", "type": "purchase_order", "fields": {"amount": "1500.00", "supplier": "Midas"}},
            {"id": "doc3", "text": "INV 1", "type": "invoice", "fields": {"amount": "2200", "supplier": "Superstore"}},
        ]
        service.add_documents(docs)

        # "Ansell"
        res = service.search("Ansell")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], "doc1")
        
        # "ansell"
        res = service.search("ansell")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], "doc1")
        
        # "from Midas"
        res = service.search("from Midas")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], "doc2")
        
        # "supplier Superstore"
        res = service.search("supplier Superstore")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], "doc3")
        
        # "Ansell below 4000"
        res = service.search("Ansell below 4000")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], "doc1")
        
        # "Midas above 3000"
        res = service.search("Midas above 3000")
        self.assertEqual(len(res), 0)
        
        # "purchase orders from Ansell"
        res = service.search("purchase orders from Ansell")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], "doc1")
        
        # "Ansell office gloves" -> fallback to FAISS, should match Ansell document
        res = service.search("Ansell office gloves")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], "doc1")
        
        # "office equipment" -> pure semantic query
        res = service.search("office equipment")
        self.assertEqual(len(res), 3)
        
    def test_ambiguity(self):
        service = SemanticSearchService(embedding_service=MockEmbeddingService())
        docs = [
            {"id": "doc1", "text": "PO 1", "type": "purchase_order", "fields": {"amount": "3800.00", "supplier": "ABC Trading"}},
            {"id": "doc2", "text": "PO 2", "type": "purchase_order", "fields": {"amount": "1500.00", "supplier": "ABC Engineering"}},
        ]
        service.add_documents(docs)
        
        # "ABC Trading"
        res = service.search("ABC Trading")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], "doc1")
        
        # "ABC" -> shouldn't match anything directly since there is no "ABC" supplier. 
        res = service.search("ABC")
        self.assertEqual(len(res), 2)

if __name__ == '__main__':
    unittest.main()
