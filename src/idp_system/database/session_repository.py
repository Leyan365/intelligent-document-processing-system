"""Opaque browser-session persistence for local authentication."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .db import APP_DB_PATH, get_connection, initialize_auth_db


SESSION_TOKEN_BYTES = 32
SESSION_TOKEN_CREATE_ATTEMPTS = 3


def create_auth_session(
    user_id: int,
    expires_at: datetime | str,
    db_path: str | Path = APP_DB_PATH,
) -> str:
    """Create a database session and return its one-time raw browser token."""
    initialize_auth_db(db_path)
    created_at = _utc_now()
    expiry = _as_utc_iso(expires_at)

    for _ in range(SESSION_TOKEN_CREATE_ATTEMPTS):
        token = secrets.token_urlsafe(SESSION_TOKEN_BYTES)
        try:
            with get_connection(db_path) as connection:
                connection.execute(
                    """
                    INSERT INTO auth_sessions (
                        user_id, token_hash, created_at, expires_at, last_seen_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, _hash_token(token), created_at, expiry, created_at),
                )
        except sqlite3.IntegrityError:
            continue
        return token

    raise RuntimeError("Could not create a unique authentication session token.")


def get_user_for_session_token(
    token: str | None,
    db_path: str | Path = APP_DB_PATH,
) -> dict[str, Any] | None:
    """Return the session user only when the token is valid and active."""
    if not token:
        return None

    now = _utc_now()
    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT u.user_id, u.username, u.email, u.created_at, u.last_login_at
            FROM auth_sessions AS s
            JOIN users AS u ON u.user_id = s.user_id
            WHERE s.token_hash = ?
              AND s.revoked_at IS NULL
              AND s.expires_at > ?
            """,
            (_hash_token(token), now),
        ).fetchone()
    return dict(row) if row is not None else None


def touch_auth_session(
    token: str | None,
    db_path: str | Path = APP_DB_PATH,
) -> bool:
    """Update last-seen time for an active session token."""
    if not token:
        return False

    now = _utc_now()
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            UPDATE auth_sessions
            SET last_seen_at = ?
            WHERE token_hash = ?
              AND revoked_at IS NULL
              AND expires_at > ?
            """,
            (now, _hash_token(token), now),
        )
    return cursor.rowcount == 1


def revoke_auth_session(
    token: str | None,
    db_path: str | Path = APP_DB_PATH,
) -> bool:
    """Revoke an active session token, making future restoration fail."""
    if not token:
        return False

    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            UPDATE auth_sessions
            SET revoked_at = ?
            WHERE token_hash = ? AND revoked_at IS NULL
            """,
            (_utc_now(), _hash_token(token)),
        )
    return cursor.rowcount == 1


def delete_expired_sessions(
    db_path: str | Path = APP_DB_PATH,
) -> int:
    """Delete expired sessions and return the number removed."""
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            "DELETE FROM auth_sessions WHERE expires_at <= ?",
            (_utc_now(),),
        )
    return max(cursor.rowcount, 0)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _as_utc_iso(value: datetime | str) -> str:
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
