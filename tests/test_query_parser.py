import unittest
from idp_system.pipeline.query_parser import parse_query, parse_amount
from decimal import Decimal

class TestQueryParser(unittest.TestCase):
    def test_amount_parsing(self):
        self.assertEqual(parse_amount("3000"), Decimal("3000"))
        self.assertEqual(parse_amount("3,000"), Decimal("3000"))
        self.assertEqual(parse_amount("3000.00"), Decimal("3000.00"))
        self.assertEqual(parse_amount("$1,500.50"), Decimal("1500.50"))
        self.assertEqual(parse_amount("Rs. 3,800.00"), Decimal("3800.00"))
        self.assertEqual(parse_amount("LKR 1500"), Decimal("1500"))

    def test_query_parser(self):
        q = parse_query("purchase orders below 3000 office supplies")
        self.assertEqual(q.document_type, "purchase_order")
        self.assertEqual(q.amount_lt, Decimal("3000"))
        self.assertEqual(q.semantic_text, "office supplies")
        
        q2 = parse_query("invoices over 5000")
        self.assertEqual(q2.document_type, "invoice")
        self.assertEqual(q2.amount_gt, Decimal("5000"))
        self.assertEqual(q2.semantic_text, "")

        q3 = parse_query("between 1000 and 3000")
        self.assertEqual(q3.amount_min, Decimal("1000"))
        self.assertEqual(q3.amount_max, Decimal("3000"))

        q4 = parse_query("amount 3000")
        self.assertEqual(q4.amount_eq, Decimal("3000"))

if __name__ == '__main__':
    unittest.main()
