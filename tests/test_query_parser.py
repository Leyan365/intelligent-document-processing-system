import unittest
from datetime import date
from decimal import Decimal

from idp_system.pipeline.query_parser import parse_date, parse_query, parse_amount

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

    def test_date_parsing_and_date_filters(self):
        self.assertEqual(parse_date("1.4.2026"), date(2026, 4, 1))
        self.assertEqual(parse_date("1st of January 2026"), date(2026, 1, 1))
        self.assertEqual(parse_date("25-Nov-2025"), date(2025, 11, 25))

        after_query = parse_query("purchase orders after 1st of January 2026")
        self.assertEqual(after_query.document_type, "purchase_order")
        self.assertEqual(after_query.date_gt, date(2026, 1, 1))
        self.assertEqual(after_query.semantic_text, "")

        range_query = parse_query("between 1.1.2026 and 31.1.2026")
        self.assertEqual(range_query.date_min, date(2026, 1, 1))
        self.assertEqual(range_query.date_max, date(2026, 1, 31))

        incomplete_query = parse_query("after January 2026")
        self.assertIsNotNone(incomplete_query.date_error)

    def test_document_number_is_preserved_after_type_parsing(self):
        invoice_query = parse_query("invoice 39519 Aaron Bergman")
        self.assertEqual("invoice", invoice_query.document_type)
        self.assertEqual("39519", invoice_query.document_number)
        self.assertEqual("Aaron Bergman", invoice_query.semantic_text)

        po_query = parse_query("PO number 5380034300")
        self.assertEqual("purchase_order", po_query.document_type)
        self.assertEqual("5380034300", po_query.document_number)
        self.assertEqual("", po_query.semantic_text)

if __name__ == '__main__':
    unittest.main()
