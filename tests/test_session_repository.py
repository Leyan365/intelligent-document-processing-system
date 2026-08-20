"""Focused tests for opaque authentication sessions."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from idp_system.database.auth_repository import AuthRepository
from idp_system.database.db import get_connection, initialize_auth_db
from idp_system.database.session_repository import (
    create_auth_session,
    delete_expired_sessions,
    get_user_for_session_token,
    revoke_auth_session,
    touch_auth_session,
)


class AuthSessionRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path("C:/tmp"))
        self.db_path = Path(self.temp_dir.name) / "sessions.db"
        initialize_auth_db(self.db_path)
        result = AuthRepository(self.db_path).create_user(
            username="session-user",
            email="session@example.test",
            password_hash="test-hash",
            salt="test-salt",
            created_at=_now().isoformat(timespec="seconds"),
        )
        self.assertTrue(result.success)
        self.user_id = int(result.user["user_id"])

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_session_creation_stores_hash_not_raw_token(self) -> None:
        token = create_auth_session(self.user_id, _now() + timedelta(hours=1), self.db_path)

        with get_connection(self.db_path) as connection:
            row = connection.execute(
                "SELECT token_hash FROM auth_sessions"
            ).fetchone()

        self.assertIsNotNone(row)
        self.assertNotEqual(token, row["token_hash"])
        self.assertEqual(
            hashlib.sha256(token.encode("utf-8")).hexdigest(),
            row["token_hash"],
        )

    def test_valid_token_restores_correct_user_and_can_be_touched(self) -> None:
        token = create_auth_session(self.user_id, _now() + timedelta(hours=1), self.db_path)

        user = get_user_for_session_token(token, self.db_path)

        self.assertIsNotNone(user)
        self.assertEqual(self.user_id, user["user_id"])
        self.assertEqual("session-user", user["username"])
        self.assertTrue(touch_auth_session(token, self.db_path))

    def test_invalid_token_is_rejected(self) -> None:
        self.assertIsNone(
            get_user_for_session_token("not-a-real-session-token", self.db_path)
        )

    def test_expired_token_is_rejected_and_cleanup_deletes_it(self) -> None:
        token = create_auth_session(self.user_id, _now() - timedelta(seconds=1), self.db_path)

        self.assertIsNone(get_user_for_session_token(token, self.db_path))
        self.assertEqual(1, delete_expired_sessions(self.db_path))

    def test_revoked_token_is_rejected(self) -> None:
        token = create_auth_session(self.user_id, _now() + timedelta(hours=1), self.db_path)

        self.assertTrue(revoke_auth_session(token, self.db_path))
        self.assertIsNone(get_user_for_session_token(token, self.db_path))

    def test_logout_revocation_is_idempotently_enforced(self) -> None:
        token = create_auth_session(self.user_id, _now() + timedelta(hours=1), self.db_path)

        self.assertTrue(revoke_auth_session(token, self.db_path))
        self.assertFalse(revoke_auth_session(token, self.db_path))
        self.assertFalse(touch_auth_session(token, self.db_path))


def _now() -> datetime:
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    unittest.main()
