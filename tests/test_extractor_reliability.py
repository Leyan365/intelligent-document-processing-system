"""Focused regression tests for business-document field extraction."""

from __future__ import annotations

import unittest

from idp_system.pipeline.extractor import extract_fields


class ExtractionReliabilityTests(unittest.TestCase):
    def test_dollar_comma_amount_is_preserved(self) -> None:
        fields = extract_fields("Total Amount $1,250.00", "invoice")
        self.assertEqual("$1,250.00", fields["amount"])

    def test_rupee_comma_amount_is_preserved(self) -> None:
        fields = extract_fields("TOTAL AMT PAYABLE Rs. 13,500.00", "receipt")
        self.assertEqual("Rs. 13,500.00", fields["amount"])

    def test_decimal_amount_is_extracted(self) -> None:
        fields = extract_fields("Grand Total 5746.60", "purchase_order")
        self.assertEqual("5746.60", fields["amount"])

    def test_short_prefixed_invoice_number_is_supported(self) -> None:
        fields = extract_fields("Invoice No INV-42", "invoice")
        self.assertEqual("INV-42", fields["invoice_number"])

    def test_unprefixed_two_digit_invoice_number_is_not_inferred(self) -> None:
        fields = extract_fields("Invoice No 42", "invoice")
        self.assertIsNone(fields["invoice_number"])

    def test_generic_receipt_heading_yields_to_real_merchant(self) -> None:
        text = "SALES RECEIPT\nQuantum Logic Solutions\nCash Total 45.50"
        fields = extract_fields(text, "receipt")
        self.assertEqual("Quantum Logic Solutions", fields["supplier"])

    def test_po_to_recipient_layout(self) -> None:
        text = (
            "PURCHASE ORDER\nPO NO : IN202502742\nTo\n"
            ": SCREENLINE (Pvt) LTD,18/4 THALWATTHA\nGrand Total 4,760.54"
        )
        fields = extract_fields(text, "purchase_order")
        self.assertEqual("SCREENLINE (Pvt) LTD", fields["supplier"])
        self.assertEqual("IN202502742", fields["invoice_number"])

    def test_po_multiline_supplier_stops_before_currency(self) -> None:
        text = (
            "PURCHASE ORDER IMPORTS\nSupplier:\nScreenline (Pvt) Ltd\nCurrency:\nUSD\n"
            "P.O. Number / Date / Version:\n1000583947 / 26.12.2025 / 1"
        )
        fields = extract_fields(text, "purchase_order")
        self.assertEqual("Screenline (Pvt) Ltd", fields["supplier"])
        self.assertEqual("1000583947", fields["invoice_number"])

    def test_po_supplier_code_layout(self) -> None:
        text = (
            "ORIGINAL\n5380034300\n25.01.2026\nPO Number :\nPO Creation Date :\n"
            "SUPPLIER:\nSCREENLINE (PVT) LTD-1007037\nAddress: Gonawala"
        )
        fields = extract_fields(text, "purchase_order")
        self.assertEqual("SCREENLINE (PVT) LTD", fields["supplier"])
        self.assertEqual("5380034300", fields["invoice_number"])

    def test_order_number_layout(self) -> None:
        fields = extract_fields("Order Number\nPO10042153\nOrder Date\n21-Jan-2026", "purchase_order")
        self.assertEqual("PO10042153", fields["invoice_number"])

    def test_missing_po_supplier_remains_missing(self) -> None:
        text = "PURCHASE ORDER\nBuyer: Ansell Lanka\nDelivery Address: Biyagama"
        fields = extract_fields(text, "purchase_order")
        self.assertIsNone(fields["supplier"])


if __name__ == "__main__":
    unittest.main()
