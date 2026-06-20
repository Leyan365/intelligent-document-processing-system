"""Local authentication service for the Streamlit application."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .database.auth_repository import AuthRepository
from .database.db import APP_DB_PATH, initialize_auth_db as _initialize_auth_db


PBKDF2_ITERATIONS = 200_000
SALT_BYTES = 32


@dataclass(frozen=True, slots=True)
class AuthResult:
    """Result returned by authentication service actions."""

    success: bool
    user: dict[str, Any] | None = None
    error: str | None = None


def initialize_auth_db(db_path: str | Path = APP_DB_PATH) -> None:
    """Initialize the local authentication database."""
    _initialize_auth_db(db_path)


def register_user(
    username: str,
    email: str | None,
    password: str,
    db_path: str | Path = APP_DB_PATH,
) -> AuthResult:
    """Register a local user with a salted PBKDF2 password hash."""
    normalized_username = username.strip()
    normalized_email = _normalize_email(email)

    validation_error = _validate_registration(normalized_username, password)
    if validation_error is not None:
        return AuthResult(False, error=validation_error)

    salt = secrets.token_bytes(SALT_BYTES)
    password_hash = _hash_password(password, salt)
    created_at = _utc_now()
    result = AuthRepository(db_path).create_user(
        username=normalized_username,
        email=normalized_email,
        password_hash=password_hash,
        salt=salt.hex(),
        created_at=created_at,
    )

    return AuthResult(result.success, user=result.user, error=result.error)


def authenticate_user(
    username_or_email: str,
    password: str,
    db_path: str | Path = APP_DB_PATH,
) -> AuthResult:
    """Authenticate a user by username or email."""
    lookup_value = username_or_email.strip()
    if not lookup_value or not password:
        return AuthResult(False, error="Username/email and password are required.")

    repository = AuthRepository(db_path)
    user = repository.get_user_by_username_or_email(lookup_value)
    if user is None:
        return AuthResult(False, error="Invalid username/email or password.")

    try:
        salt = bytes.fromhex(str(user["salt"]))
        expected_hash = str(user["password_hash"])
    except (KeyError, ValueError, TypeError):
        return AuthResult(False, error="Invalid username/email or password.")

    supplied_hash = _hash_password(password, salt)
    if not hmac.compare_digest(supplied_hash, expected_hash):
        return AuthResult(False, error="Invalid username/email or password.")

    update_last_login(int(user["user_id"]), db_path=db_path)
    user = repository.get_user_by_id(int(user["user_id"])) or user
    return AuthResult(True, user=_public_user(user))


def get_user_by_id(user_id: int, db_path: str | Path = APP_DB_PATH) -> dict[str, Any] | None:
    """Return a public user record by ID."""
    user = AuthRepository(db_path).get_user_by_id(user_id)
    return _public_user(user) if user is not None else None


def update_last_login(user_id: int, db_path: str | Path = APP_DB_PATH) -> None:
    """Store the current UTC time as the user's last login timestamp."""
    AuthRepository(db_path).update_last_login(user_id, _utc_now())


def _validate_registration(username: str, password: str) -> str | None:
    if not username:
        return "Username is required."
    if not password:
        return "Password is required."
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    return None


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    ).hex()


def _normalize_email(email: str | None) -> str | None:
    if email is None:
        return None
    cleaned = email.strip().lower()
    return cleaned or None


def _public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": user.get("user_id"),
        "username": user.get("username"),
        "email": user.get("email"),
        "created_at": user.get("created_at"),
        "last_login_at": user.get("last_login_at"),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
