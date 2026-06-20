"""User authentication persistence for the local application database."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import APP_DB_PATH, get_connection


@dataclass(frozen=True, slots=True)
class RepositoryResult:
    """Result returned by write operations in the auth repository."""

    success: bool
    user: dict[str, Any] | None = None
    error: str | None = None


class AuthRepository:
    """SQLite-backed user repository with SQL isolated for future migration."""

    def __init__(self, db_path: str | Path = APP_DB_PATH) -> None:
        self.db_path = Path(db_path)

    def create_user(
        self,
        username: str,
        email: str | None,
        password_hash: str,
        salt: str,
        created_at: str,
    ) -> RepositoryResult:
        try:
            with get_connection(self.db_path) as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO users (username, email, password_hash, salt, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (username, email, password_hash, salt, created_at),
                )
                user_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            existing = self.get_user_by_username_or_email(username)
            if existing is not None:
                return RepositoryResult(False, error="Username already exists.")
            if email:
                existing = self.get_user_by_username_or_email(email)
                if existing is not None:
                    return RepositoryResult(False, error="Email already exists.")
            return RepositoryResult(False, error=f"Could not create user: {exc}")

        return RepositoryResult(
            True,
            user={
                "user_id": user_id,
                "username": username,
                "email": email,
                "created_at": created_at,
                "last_login_at": None,
            },
        )

    def get_user_by_username_or_email(self, username_or_email: str) -> dict[str, Any] | None:
        with get_connection(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT user_id, username, email, password_hash, salt, created_at, last_login_at
                FROM users
                WHERE username = ? OR email = ?
                """,
                (username_or_email, username_or_email),
            ).fetchone()
        return _row_to_dict(row)

    def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
        with get_connection(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT user_id, username, email, password_hash, salt, created_at, last_login_at
                FROM users
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        return _row_to_dict(row)

    def update_last_login(self, user_id: int, last_login_at: str) -> None:
        with get_connection(self.db_path) as connection:
            connection.execute(
                "UPDATE users SET last_login_at = ? WHERE user_id = ?",
                (last_login_at, user_id),
            )


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)
