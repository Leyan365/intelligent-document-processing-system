"""Ownership checks for persisted current-document loading."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from idp_system.database.auth_repository import AuthRepository
from idp_system.database.db import initialize_auth_db, initialize_document_tables
from idp_system.database.document_repository import (
    count_documents_for_user,
    get_document_by_id_for_user,
    list_documents_for_user_page,
    save_processed_document,
)


class DocumentOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path("C:/tmp"))
        self.db_path = Path(self.temp_dir.name) / "documents.db"
        initialize_auth_db(self.db_path)
        initialize_document_tables(self.db_path)
        repository = AuthRepository(self.db_path)
        self.owner_id = self._create_user(repository, "owner")
        self.other_user_id = self._create_user(repository, "other")
        record = save_processed_document(
            user_id=self.owner_id,
            uploaded_file_metadata={
                "original_filename": "owned.pdf",
                "file_size": 3,
                "content_type": "application/pdf",
            },
            file_hash="owned-file-hash",
            stored_path=Path(self.temp_dir.name) / "owned.pdf",
            result={
                "text": "owned document",
                "type": "invoice",
                "fields": {"supplier": "Private Supplier"},
                "validation": {"pipeline_status": "processed"},
            },
            db_path=self.db_path,
        )
        self.document_id = int(record["document_id"])

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_owner_can_load_current_document(self) -> None:
        record = get_document_by_id_for_user(
            self.document_id,
            self.owner_id,
            self.db_path,
        )
        self.assertIsNotNone(record)
        self.assertEqual("owned.pdf", record["original_filename"])

    def test_other_user_cannot_load_current_document(self) -> None:
        record = get_document_by_id_for_user(
            self.document_id,
            self.other_user_id,
            self.db_path,
        )
        self.assertIsNone(record)

    def test_paginated_filename_search_remains_scoped_to_owner(self) -> None:
        for index, filename in enumerate(
            ["invoice-001.pdf", "invoice_100%.pdf", "receipt.pdf", "Invoice-003.pdf"],
            start=1,
        ):
            self._save_document(self.owner_id, filename, f"owner-search-{index}")
        self._save_document(self.other_user_id, "invoice-private.pdf", "other-user-search")

        self.assertEqual(3, count_documents_for_user(self.owner_id, "invoice", self.db_path))
        self.assertEqual(1, count_documents_for_user(self.owner_id, "100%", self.db_path))

        first_page = list_documents_for_user_page(
            self.owner_id,
            page_size=2,
            filename_query="invoice",
            db_path=self.db_path,
        )
        second_page = list_documents_for_user_page(
            self.owner_id,
            page_size=2,
            offset=2,
            filename_query="invoice",
            db_path=self.db_path,
        )

        self.assertEqual(["Invoice-003.pdf", "invoice_100%.pdf"], [row["original_filename"] for row in first_page])
        self.assertEqual(["invoice-001.pdf"], [row["original_filename"] for row in second_page])

    def _save_document(self, user_id: int, filename: str, file_hash: str) -> None:
        save_processed_document(
            user_id=user_id,
            uploaded_file_metadata={"original_filename": filename, "file_size": 3},
            file_hash=file_hash,
            stored_path=Path(self.temp_dir.name) / filename,
            result={"text": filename, "type": "invoice", "fields": {}, "validation": {}},
            db_path=self.db_path,
        )

    def _create_user(self, repository: AuthRepository, username: str) -> int:
        created = repository.create_user(
            username=username,
            email=None,
            password_hash="test-hash",
            salt="test-salt",
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.assertTrue(created.success)
        return int(created.user["user_id"])


if __name__ == "__main__":
    unittest.main()
