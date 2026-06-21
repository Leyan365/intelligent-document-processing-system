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
    connection.execute("PRAGMA foreign_keys = ON")
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


def initialize_document_tables(db_path: str | Path = APP_DB_PATH) -> None:
    """Create document persistence tables if they do not exist."""
    with get_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                file_hash TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                stored_filename TEXT,
                stored_path TEXT,
                file_size INTEGER,
                document_type TEXT,
                processing_status TEXT,
                raw_text TEXT,
                text_preview TEXT,
                result_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                UNIQUE(user_id, file_hash),
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS classifications (
                classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                label TEXT,
                confidence REAL,
                confidence_source TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS extracted_fields (
                field_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                field_name TEXT NOT NULL,
                field_value TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS validation_results (
                validation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                pipeline_status TEXT,
                validation_score REAL,
                warning_count INTEGER,
                critical_warning_count INTEGER,
                warnings_json TEXT,
                validation_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            )
            """
        )


class Database:
    """Small compatibility wrapper around the local SQLite connection."""

    def __init__(self, db_path: str | Path = APP_DB_PATH) -> None:
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        return get_connection(self.db_path)
