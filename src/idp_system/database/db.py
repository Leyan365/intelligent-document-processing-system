"""SQLite database helpers for local application storage."""

from __future__ import annotations

import sqlite3
from pathlib import Path


APP_DB_PATH = Path("data/app/idp_app.db")


def get_connection(db_path: str | Path = APP_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection for the local application database."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_auth_db(db_path: str | Path = APP_DB_PATH) -> None:
    """Create the local authentication schema if it does not exist."""
    with get_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login_at TEXT
            )
            """
        )


class Database:
    """Small compatibility wrapper around the local SQLite connection."""

    def __init__(self, db_path: str | Path = APP_DB_PATH) -> None:
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        return get_connection(self.db_path)
