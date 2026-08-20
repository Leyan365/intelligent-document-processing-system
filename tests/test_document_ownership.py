"""Ownership checks for persisted current-document loading."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from idp_system.database.auth_repository import AuthRepository
from idp_system.database.db import initialize_auth_db, initialize_document_tables
from idp_system.database.document_repository import (
    get_document_by_id_for_user,
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
