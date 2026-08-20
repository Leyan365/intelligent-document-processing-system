"""Database integration package."""

from .db import (
    APP_DB_PATH,
    Database,
    get_connection,
    initialize_auth_db,
    initialize_document_tables,
)
from .session_repository import (
    create_auth_session,
    delete_expired_sessions,
    get_user_for_session_token,
    revoke_auth_session,
    touch_auth_session,
)

__all__ = [
    "APP_DB_PATH",
    "Database",
    "get_connection",
    "initialize_auth_db",
    "initialize_document_tables",
    "create_auth_session",
    "delete_expired_sessions",
    "get_user_for_session_token",
    "revoke_auth_session",
    "touch_auth_session",
]
