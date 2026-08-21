"""Persistent document storage for processed IDP results."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .db import APP_DB_PATH, get_connection, initialize_document_tables as _initialize_document_tables


TEXT_PREVIEW_CHARS = 1000


def initialize_document_tables(db_path: str | Path = APP_DB_PATH) -> None:
    """Initialize document persistence tables."""
    _initialize_document_tables(db_path)


def compute_file_hash(file_bytes: bytes) -> str:
    """Return a SHA-256 hex digest for uploaded file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def get_document_by_user_and_hash(
    user_id: int,
    file_hash: str,
    db_path: str | Path = APP_DB_PATH,
) -> dict[str, Any] | None:
    """Return one persisted document for a user and file hash."""
    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM documents
            WHERE user_id = ? AND file_hash = ?
            """,
            (user_id, file_hash),
        ).fetchone()
    return _document_from_row(row)


def get_document_by_id_for_user(
    document_id: int,
    user_id: int,
    db_path: str | Path = APP_DB_PATH,
) -> dict[str, Any] | None:
    """Return one persisted document only if it belongs to the user."""
    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM documents
            WHERE document_id = ? AND user_id = ?
            """,
            (document_id, user_id),
        ).fetchone()
    return _document_from_row(row)


def list_documents_for_user(
    user_id: int,
    db_path: str | Path = APP_DB_PATH,
) -> list[dict[str, Any]]:
    """List persisted documents for a user, newest first."""
    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM documents
            WHERE user_id = ?
            ORDER BY created_at DESC, document_id DESC
            """,
            (user_id,),
        ).fetchall()
    return [_document_from_row(row) for row in rows]


def count_documents_for_user(
    user_id: int,
    filename_query: str | None = None,
    db_path: str | Path = APP_DB_PATH,
) -> int:
    """Count a user's documents, optionally filtered by a filename fragment."""
    where_clause, parameters = _document_filename_filter(user_id, filename_query)
    with get_connection(db_path) as connection:
        row = connection.execute(
            f"SELECT COUNT(*) AS document_count FROM documents {where_clause}",
            parameters,
        ).fetchone()
    return int(row["document_count"] if row is not None else 0)


def list_documents_for_user_page(
    user_id: int,
    *,
    page_size: int = 20,
    offset: int = 0,
    filename_query: str | None = None,
    db_path: str | Path = APP_DB_PATH,
) -> list[dict[str, Any]]:
    """Return one newest-first, ownership-scoped page of document records."""
    if page_size < 1:
        raise ValueError("page_size must be at least 1.")
    if offset < 0:
        raise ValueError("offset cannot be negative.")

    where_clause, parameters = _document_filename_filter(user_id, filename_query)
    with get_connection(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT *
            FROM documents
            {where_clause}
            ORDER BY created_at DESC, document_id DESC
            LIMIT ? OFFSET ?
            """,
            (*parameters, page_size, offset),
        ).fetchall()
    return [_document_from_row(row) for row in rows]


def save_processed_document(
    user_id: int,
    uploaded_file_metadata: dict[str, Any],
    file_hash: str,
    stored_path: str | Path,
    result: dict[str, Any],
    db_path: str | Path = APP_DB_PATH,
) -> dict[str, Any]:
    """Persist one successfully processed document result snapshot."""
    initialize_document_tables(db_path)
    now = _utc_now()
    stored = Path(stored_path)
    original_filename = str(uploaded_file_metadata.get("original_filename") or stored.name)
    file_size = _safe_int(uploaded_file_metadata.get("file_size"))
    text = str(result.get("text") or "")
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    fields = result.get("fields") if isinstance(result.get("fields"), dict) else {}
    document_type = _none_or_str(result.get("type"))
    result_json = json.dumps(_json_safe(result), ensure_ascii=True, sort_keys=True)

    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO documents (
                user_id, file_hash, original_filename, stored_filename, stored_path,
                file_size, document_type, processing_status, raw_text, text_preview,
                result_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                file_hash,
                original_filename,
                stored.name,
                str(stored),
                file_size,
                document_type,
                "completed",
                text,
                _text_preview(text),
                result_json,
                now,
                now,
            ),
        )
        document_id = int(cursor.lastrowid)
        connection.execute(
            """
            INSERT INTO classifications (
                document_id, label, confidence, confidence_source, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                document_id,
                document_type,
                _safe_float(result.get("confidence")),
                _none_or_str(result.get("confidence_source")),
                now,
            ),
        )
        for field_name, field_value in fields.items():
            connection.execute(
                """
                INSERT INTO extracted_fields (document_id, field_name, field_value, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (document_id, str(field_name), _none_or_str(field_value), now),
            )
        connection.execute(
            """
            INSERT INTO validation_results (
                document_id, pipeline_status, validation_score, warning_count,
                critical_warning_count, warnings_json, validation_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                _none_or_str(validation.get("pipeline_status")),
                _safe_float(validation.get("validation_score")),
                _safe_int(validation.get("total_warnings")),
                _safe_int(validation.get("critical_warning_count")),
                json.dumps(_json_safe(validation.get("warnings", [])), ensure_ascii=True),
                json.dumps(_json_safe(validation), ensure_ascii=True, sort_keys=True),
                now,
            ),
        )

    saved = get_document_by_id_for_user(document_id, user_id, db_path)
    if saved is None:
        raise RuntimeError("Saved document could not be loaded from the database.")
    return saved


def _document_from_row(row: Any | None) -> dict[str, Any] | None:
    if row is None:
        return None
    document = dict(row)
    result = _load_result_json(document.get("result_json"))
    document["result"] = result
    return document


def _document_filename_filter(user_id: int, filename_query: str | None) -> tuple[str, tuple[Any, ...]]:
    """Build an ownership filter and an optional literal filename fragment filter."""
    normalized_query = (filename_query or "").strip()
    if not normalized_query:
        return "WHERE user_id = ?", (user_id,)

    escaped_query = (
        normalized_query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    )
    return (
        "WHERE user_id = ? AND original_filename LIKE ? ESCAPE '\\' COLLATE NOCASE",
        (user_id, f"%{escaped_query}%"),
    )


def _load_result_json(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _text_preview(text: str) -> str:
    return " ".join(text.split())[:TEXT_PREVIEW_CHARS]


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _none_or_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
