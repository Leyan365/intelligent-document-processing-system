"""Small offline tests for previously uncovered core pipeline behavior."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import fitz

from idp_system.core.models import Document, DocumentType
from idp_system.database.auth_repository import AuthRepository
from idp_system.database.db import initialize_auth_db, initialize_document_tables
from idp_system.database.document_repository import save_processed_document
from idp_system.pipeline.classifier import DocumentClassifier
from idp_system.pipeline.loader import LocalTextExtractionLoader
from idp_system.pipeline.validation import validate_pipeline
from idp_system.system import IDPSystem
from idp_system.ui.streamlit_app import _cleanup_unpersisted_upload


class FakeLoader:
    def load(self, source: str | Path) -> Document:
        return Document(
            title="controlled",
            content=(
                "Invoice No INV-42\nSupplier: Acme Office Supplies\n"
                "Invoice Date: 2026-08-20\nTotal Amount $1,250.00"
            ),
            source=str(source),
            doc_type=DocumentType.PDF,
            extraction_method="pymupdf",
        )


class FakeClassifier:
    def classify_with_confidence(self, text: str) -> dict[str, object]:
        return {"label": "invoice", "confidence": 0.98, "confidence_source": "controlled"}


class FakeSearchService:
    def __init__(self) -> None:
        self.documents: list[dict[str, object]] = []

    def add_documents(self, documents: list[dict[str, object]]) -> None:
        self.documents.extend(documents)

    def search(self, query: str, k: int = 5) -> list[dict[str, object]]:
        return self.documents[:k]


class CorePipelineTests(unittest.TestCase):
    def test_native_pdf_text_extraction_avoids_ocr(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path("C:/tmp")) as temp_dir:
            pdf_path = Path(temp_dir) / "native.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_text((72, 72), "Native invoice text INV-42 Total Amount 1250.00")
            document.save(pdf_path)
            document.close()

            loaded = LocalTextExtractionLoader(min_pdf_text_chars=10).load(pdf_path)

        self.assertIn("Native invoice text", loaded.content)
        self.assertEqual("pymupdf", loaded.extraction_method)

    def test_controlled_classifier_uses_invoice_signal(self) -> None:
        result = DocumentClassifier().classify_with_confidence(
            "INVOICE Invoice Number INV-900 Balance Due 1250.00"
        )
        self.assertEqual("invoice", result["label"])

    def test_validation_accepts_complete_controlled_result(self) -> None:
        result = validate_pipeline(
            text="Invoice business document with sufficient readable text and complete fields. " * 3,
            metadata={"extraction_method": "pymupdf"},
            document_type="invoice",
            classification_confidence=0.98,
            confidence_source="model",
            fields={
                "invoice_number": "INV-42",
                "date": "2026-08-20",
                "amount": "$1,250.00",
                "supplier": "Acme Office Supplies",
            },
        )
        self.assertEqual("processed", result["pipeline_status"])

    def test_integrated_processing_path(self) -> None:
        search = FakeSearchService()
        system = IDPSystem(loader=FakeLoader(), classifier=FakeClassifier(), search_service=search)

        result = system.process_document("controlled.pdf")

        self.assertEqual("invoice", result["type"])
        self.assertEqual("INV-42", result["fields"]["invoice_number"])
        self.assertEqual("$1,250.00", result["fields"]["amount"])
        self.assertEqual(1, len(search.documents))

    def test_duplicate_file_hash_is_rejected_per_user(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path("C:/tmp")) as temp_dir:
            db_path = Path(temp_dir) / "duplicates.db"
            initialize_auth_db(db_path)
            initialize_document_tables(db_path)
            created = AuthRepository(db_path).create_user(
                username="duplicate-user",
                email=None,
                password_hash="hash",
                salt="salt",
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            user_id = int(created.user["user_id"])
            arguments = {
                "user_id": user_id,
                "uploaded_file_metadata": {"original_filename": "invoice.pdf", "file_size": 3},
                "file_hash": "same-hash",
                "stored_path": Path(temp_dir) / "invoice.pdf",
                "result": {"text": "invoice", "type": "invoice", "fields": {}, "validation": {}},
                "db_path": db_path,
            }
            save_processed_document(**arguments)
            with self.assertRaises(sqlite3.IntegrityError):
                save_processed_document(**arguments)

    def test_unpersisted_upload_is_removed_after_failure(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path("C:/tmp")) as temp_dir:
            path = Path(temp_dir) / "new-upload.pdf"
            path.write_bytes(b"new")
            with patch(
                "idp_system.ui.streamlit_app.get_document_by_user_and_hash",
                return_value=None,
            ):
                error = _cleanup_unpersisted_upload(path, 1, "hash")
            self.assertIsNone(error)
            self.assertFalse(path.exists())

    def test_persisted_original_is_never_removed(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path("C:/tmp")) as temp_dir:
            path = Path(temp_dir) / "persisted.pdf"
            path.write_bytes(b"persisted")
            with patch(
                "idp_system.ui.streamlit_app.get_document_by_user_and_hash",
                return_value={"stored_path": str(path)},
            ):
                error = _cleanup_unpersisted_upload(path, 1, "hash")
            self.assertIsNone(error)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
