"""Streamlit dashboard shell for the local IDP pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from idp_system.auth import authenticate_user, initialize_auth_db, register_user
from idp_system.database.document_repository import (
    compute_file_hash,
    get_document_by_user_and_hash,
    initialize_document_tables,
    list_documents_for_user,
    save_processed_document,
)
from idp_system.system import IDPSystem


SUPPORTED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg"]
UPLOAD_STORAGE_ROOT = Path("data/app/uploads")
PREVIEW_CHARS = 1000


def main() -> None:
    st.set_page_config(
        page_title="Intelligent Document Processing System",
        layout="wide",
    )
    initialize_auth_db()
    initialize_document_tables()
    _ensure_session_state()

    st.title("Intelligent Document Processing System")

    if not st.session_state.authenticated:
        render_auth_page()
        return

    _ensure_active_user_state()
    _render_authenticated_sidebar()
    page = st.sidebar.radio(
        "Navigation",
        ["Upload & Process", "Search", "Document History"],
    )

    if page == "Upload & Process":
        render_upload_page()
    elif page == "Search":
        render_search_page()
    else:
        render_history_page()


def _ensure_session_state() -> None:
    if "system" not in st.session_state:
        st.session_state.system = IDPSystem()
    if "processed_history" not in st.session_state:
        st.session_state.processed_history = []
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "active_user_id" not in st.session_state:
        st.session_state.active_user_id = None
    if "search_index_user_id" not in st.session_state:
        st.session_state.search_index_user_id = None
    if "search_index_document_count" not in st.session_state:
        st.session_state.search_index_document_count = None


def _ensure_active_user_state() -> None:
    user_id = st.session_state.user_id
    if user_id is not None and st.session_state.active_user_id != user_id:
        _reset_document_session_state(user_id)


def _reset_document_session_state(user_id: int | None = None) -> None:
    st.session_state.system = IDPSystem()
    st.session_state.processed_history = []
    st.session_state.active_user_id = user_id
    st.session_state.search_index_user_id = None
    st.session_state.search_index_document_count = None


def render_auth_page() -> None:
    st.info(
        "This is a local academic prototype authentication system. "
        "It does not include enterprise-grade controls such as MFA or RBAC yet."
    )
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        render_login_form()

    with register_tab:
        render_register_form()


def render_login_form() -> None:
    st.subheader("Login")
    with st.form("login_form"):
        username_or_email = st.text_input("Username or email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")

    if not submitted:
        return

    result = authenticate_user(username_or_email, password)
    if not result.success or result.user is None:
        st.error(result.error or "Login failed.")
        return

    user_id = int(result.user["user_id"])
    st.session_state.authenticated = True
    st.session_state.user_id = user_id
    st.session_state.username = result.user["username"]
    _reset_document_session_state(user_id)
    st.success("Login successful.")
    st.rerun()


def render_register_form() -> None:
    st.subheader("Register")
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email (optional)")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Register", type="primary")

    if not submitted:
        return

    if password != confirm_password:
        st.error("Passwords do not match.")
        return

    result = register_user(username, email, password)
    if not result.success:
        st.error(result.error or "Registration failed.")
        return

    st.success("Registration successful. Please use the Login tab to sign in.")


def _render_authenticated_sidebar() -> None:
    st.sidebar.markdown(f"Signed in as **{st.session_state.username}**")
    st.sidebar.caption("Local academic prototype auth. No MFA/RBAC yet.")
    if st.sidebar.button("Logout"):
        _logout()


def _logout() -> None:
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.username = None
    _reset_document_session_state(None)
    st.rerun()


def render_upload_page() -> None:
    st.header("Upload & Process")

    with st.container(border=True):
        st.markdown("**Start with a document**")
        st.write("Upload a digital PDF, scanned PDF, or document image. The system will extract text, classify the document, extract key fields, and index it for search.")
        uploaded_file = st.file_uploader(
            "Choose a PDF or image file",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=False,
        )

    if uploaded_file is None:
        with st.container(border=True):
            st.markdown("**No document uploaded yet**")
            st.write("Upload a PDF or image to start processing.")
            st.caption("Supported formats: PDF, PNG, JPG, and JPEG.")
        return

    with st.container(border=True):
        st.markdown("**Selected File**")
        col_name, col_type, col_size = st.columns(3)
        col_name.metric("Name", uploaded_file.name)
        col_type.metric("Type", uploaded_file.type or "Unknown")
        col_size.metric("Size", _format_bytes(uploaded_file.size))

    if st.button("Process Document", type="primary", width="stretch"):
        user_id = _current_user_id()
        file_bytes = uploaded_file.getvalue()
        file_hash = compute_file_hash(file_bytes)
        duplicate = get_document_by_user_and_hash(user_id, file_hash)

        if duplicate is not None:
            result = _result_from_document_record(duplicate)
            _remember_result(result)
            st.info("Duplicate upload detected. Existing processed result was loaded.")
            render_result(result)
            return

        stored_path = _save_uploaded_file_bytes(uploaded_file, file_bytes, user_id)
        progress = st.progress(0)
        stages = st.empty()

        try:
            _render_stages(stages, active_index=0)
            progress.progress(20, text="Text Extraction")

            _render_stages(stages, active_index=1)
            progress.progress(45, text="Classification")

            _render_stages(stages, active_index=2)
            progress.progress(70, text="Information Extraction")

            result = st.session_state.system.process_document(stored_path)

            _render_stages(stages, active_index=3)
            progress.progress(90, text="Search Indexing")

            saved = save_processed_document(
                user_id=user_id,
                uploaded_file_metadata={
                    "original_filename": uploaded_file.name,
                    "file_size": len(file_bytes),
                    "content_type": uploaded_file.type,
                },
                file_hash=file_hash,
                stored_path=stored_path,
                result=result,
            )
            result = _result_from_document_record(saved)
            _remember_result(result)
            _mark_search_index_stale()
            _render_stages(stages, active_index=4)
            progress.progress(100, text="Complete")

            st.toast("Document processed and saved successfully")
            st.success("Document processed and saved successfully.")
            render_result(result)
        except Exception as exc:
            st.error("Document processing failed.")
            st.exception(exc)


def render_result(result: dict[str, Any]) -> None:
    st.subheader("Processing Result")
    left, right = st.columns([2, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("**Document Preview**")
            text = str(result.get("text", ""))
            preview = _snippet(text, PREVIEW_CHARS)
            st.text_area("Extracted text preview", value=preview, height=260, disabled=True)
            if len(text) > PREVIEW_CHARS:
                st.caption(f"Preview limited to {PREVIEW_CHARS} characters.")

    with right:
        with st.container(border=True):
            st.markdown("**Document Type**")
            _badge(_classification_label(result))

        st.markdown("**Extracted Fields**")
        field_cols = st.columns(2)
        FIELD_LABELS = {
            "invoice_number": "Invoice / Order No.",
            "date": "Date",
            "amount": "Amount",
            "supplier": "Supplier",
        }

        for index, field_name in enumerate(("invoice_number", "date", "amount", "supplier")):
            with field_cols[index % 2]:
                label = FIELD_LABELS.get(field_name, field_name.replace("_", " ").title())
                _editable_field_card(result, field_name, label)

        render_validation_section(result)


def render_validation_section(result: dict[str, Any]) -> None:
    validation = result.get("validation")
    if not isinstance(validation, dict):
        return

    with st.container(border=True):
        st.markdown("**Validation & Confidence**")
        pipeline_status = str(validation.get("pipeline_status", "processed"))
        _status_badge(_pipeline_status_label(pipeline_status), pipeline_status)

        score = validation.get("validation_score")
        total_warnings = validation.get("total_warnings", 0)
        critical_count = validation.get("critical_warning_count", 0)
        st.caption(
            f"Score: {_display_value(score)} | "
            f"Warnings: {total_warnings} | Critical: {critical_count}"
        )

        status_cols = st.columns(3)
        status_cols[0].caption(
            f"OCR quality: {_component_status(validation.get('ocr_quality'))}"
        )
        status_cols[1].caption(
            f"Classification: {_component_status(validation.get('classification'))}"
        )
        status_cols[2].caption(
            f"Fields: {_component_status(validation.get('fields'))}"
        )

        warnings = validation.get("warnings")
        if isinstance(warnings, list) and warnings:
            shown = [str(warning) for warning in warnings[:4]]
            st.caption("Warnings: " + " | ".join(shown))
            if len(warnings) > len(shown):
                st.caption(f"+ {len(warnings) - len(shown)} more")


def render_search_page() -> None:
    st.header("Search")
    st.info(
        "Semantic search retrieves relevant processed documents and snippets. "
        "It does not generate answers."
    )
    _ensure_search_index_for_current_user()

    with st.container(border=True):
        query = st.text_input(
            "Search documents",
            placeholder="Search by meaning, supplier, amount, or document type",
        )
        st.caption(
            "Try queries such as 'invoice from supplier', 'purchase order amount', "
            "or 'documents from Lalan Rubbers'."
        )
        k = st.slider("Results", min_value=1, max_value=10, value=5)

    if not query:
        st.info("Enter a query to search processed documents.")
        return

    try:
        results = st.session_state.system.search(query, k=k)
    except Exception as exc:
        st.error("Search failed.")
        st.exception(exc)
        return

    if not results:
        st.warning("No matching documents found. Try a different query.")
        return

    st.subheader("Search Results")

    for rank, result in enumerate(results, start=1):
        with st.container(border=True):
            fields = _fields_from_result(result)
            metadata = result.get("metadata") or {}
            doc_type = result.get("type") or metadata.get("type") or "unknown"
            confidence = result.get("confidence", metadata.get("confidence"))
            confidence_source = result.get("confidence_source", metadata.get("confidence_source"))

            header_left, header_right = st.columns([3, 1])
            with header_left:
                st.markdown(f"**#{rank} - {result.get('id', 'unknown')}**")
                _badge(_classification_label_text(doc_type, confidence, confidence_source))
            with header_right:
                st.metric("Similarity Score", _format_score(result.get("score")))

            st.markdown("**Key Information**")
            info_cols = st.columns(4)
            info_cols[0].markdown(
                f"**Supplier**  \n`{_display_value(fields.get('supplier'))}`"
            )
            info_cols[1].markdown(
                f"**Date**  \n`{_display_value(fields.get('date'))}`"
            )
            info_cols[2].markdown(
                f"**Amount**  \n`{_display_value(fields.get('amount'))}`"
            )
            info_cols[3].markdown(
                f"**Invoice / Order No.**  \n`{_display_value(fields.get('invoice_number'))}`"
            )

            st.markdown("**Matched Preview**")
            st.markdown(_highlight_query(_snippet(str(result.get("text", "")), 200), query))


def render_history_page() -> None:
    st.header("Document History")
    records = list_documents_for_user(_current_user_id())

    if not records:
        st.info("No processed documents yet\n\nProcess your first document to see results here.")
        return

    rows = []
    for record in records:
        result = _result_from_document_record(record)
        fields = result.get("fields") or {}
        validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
        rows.append(
            {
                "filename": record.get("original_filename"),
                "processed_at": record.get("created_at"),
                "status": _pipeline_status_label(str(validation.get("pipeline_status", "processed"))),
                "type": result.get("type"),
                "supplier": fields.get("supplier"),
                "date": fields.get("date"),
                "amount": fields.get("amount"),
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True)

    st.markdown("**Recent Documents**")
    for record in records[:5]:
        result = _result_from_document_record(record)
        fields = result.get("fields") or {}
        validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
        with st.container(border=True):
            top_left, top_right = st.columns([3, 1])
            with top_left:
                st.markdown(f"**{record.get('original_filename')}**")
                _badge(
                    _classification_label_text(
                        result.get("type"),
                        result.get("confidence"),
                        result.get("confidence_source"),
                    )
                )
                st.caption(f"Processed: {_display_value(record.get('created_at'))}")
            with top_right:
                _status_badge(
                    _pipeline_status_label(str(validation.get("pipeline_status", "processed"))),
                    str(validation.get("pipeline_status", "processed")),
                )
            cols = st.columns(3)
            cols[0].caption(f"Supplier: {_display_value(fields.get('supplier'))}")
            cols[1].caption(f"Date: {_display_value(fields.get('date'))}")
            cols[2].caption(f"Amount: {_display_value(fields.get('amount'))}")
    st.caption("History is loaded from the local database for the signed-in user.")


def _save_uploaded_file_bytes(uploaded_file: Any, file_bytes: bytes, user_id: int) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    upload_dir = UPLOAD_STORAGE_ROOT / str(user_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid4().hex}{suffix}"
    stored_path = upload_dir / safe_name
    stored_path.write_bytes(file_bytes)
    return stored_path


def _current_user_id() -> int:
    user_id = st.session_state.user_id
    if user_id is None:
        raise RuntimeError("A logged-in user is required.")
    return int(user_id)


def _result_from_document_record(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record.get("result") or {})
    if not result.get("text") and record.get("raw_text"):
        result["text"] = record.get("raw_text")
    if not result.get("type") and record.get("document_type"):
        result["type"] = record.get("document_type")
    result.setdefault("id", str(record.get("document_id")))
    result["persistent_document_id"] = record.get("document_id")
    result["source_filename"] = record.get("original_filename")
    return result


def _remember_result(result: dict[str, Any]) -> None:
    document_id = result.get("persistent_document_id") or result.get("id")
    history = st.session_state.processed_history
    if document_id is not None:
        history[:] = [
            item for item in history
            if (item.get("persistent_document_id") or item.get("id")) != document_id
        ]
    history.append(result)


def _mark_search_index_stale() -> None:
    st.session_state.search_index_user_id = None
    st.session_state.search_index_document_count = None


def _ensure_search_index_for_current_user() -> None:
    user_id = _current_user_id()
    records = list_documents_for_user(user_id)
    if (
        st.session_state.search_index_user_id == user_id
        and st.session_state.search_index_document_count == len(records)
    ):
        return

    st.session_state.system = IDPSystem()
    documents = []
    for record in reversed(records):
        result = _result_from_document_record(record)
        text = str(result.get("text") or record.get("raw_text") or "")
        if not text.strip():
            continue
        fields = result.get("fields") if isinstance(result.get("fields"), dict) else {}
        document_type = str(result.get("type") or record.get("document_type") or "unknown")
        documents.append(
            {
                "id": str(record.get("document_id")),
                "text": _build_search_text(document_type, fields, text),
                "type": document_type,
                "confidence": result.get("confidence"),
                "confidence_source": result.get("confidence_source"),
                "fields": fields,
                "source": record.get("stored_path"),
            }
        )

    if documents:
        st.session_state.system.search_service.add_documents(documents)
    st.session_state.search_index_user_id = user_id
    st.session_state.search_index_document_count = len(records)


def _build_search_text(
    document_type: str,
    fields: dict[str, Any],
    document_text: str,
    content_limit: int = 2500,
) -> str:
    clean_content = " ".join(document_text.split())[:content_limit]
    return (
        f"{document_type} document.\n"
        f"Supplier: {_field_value(fields.get('supplier'))}\n"
        f"Invoice / Order No.: {_field_value(fields.get('invoice_number'))}\n"
        f"Date: {_field_value(fields.get('date'))}\n"
        f"Amount: {_field_value(fields.get('amount'))}\n\n"
        f"Relevant Content:\n{clean_content}"
    )


def _field_value(value: Any) -> str:
    return "" if value in (None, "") else str(value)


def _render_stages(container: Any, active_index: int) -> None:
    stage_names = [
        "Text Extraction",
        "Classification",
        "Information Extraction",
        "Search Indexing",
    ]
    lines = []
    for index, stage_name in enumerate(stage_names):
        if index < active_index:
            marker = "done"
        elif index == active_index:
            marker = "active"
        else:
            marker = "pending"
        lines.append(f"[{marker}] {stage_name}")
    container.code("\n".join(lines), language="text")


def _snippet(text: str, length: int = 500) -> str:
    text = " ".join(text.split())
    return text[:length] + ("..." if len(text) > length else "")


def _editable_field_card(result: dict[str, Any], field_name: str, label: str) -> None:
    fields = result.setdefault("fields", {})
    document_id = result.get("persistent_document_id") or result.get("id", "document")
    with st.container(border=True):
        st.caption(label)
        updated_value = st.text_input(
            label,
            value="" if fields.get(field_name) is None else str(fields.get(field_name)),
            key=f"{document_id}_{field_name}",
            label_visibility="collapsed",
        )
        fields[field_name] = updated_value or None


def _badge(value: str) -> None:
    st.markdown(
        f"<span style='display:inline-block;padding:0.2rem 0.55rem;border-radius:0.35rem;"
        f"background:#eef2ff;color:#1f2937;font-weight:600;font-size:0.9rem;'>{value}</span>",
        unsafe_allow_html=True,
    )


def _status_badge(value: str, status: str) -> None:
    background, color = _status_colors(status)
    st.markdown(
        f"<span style='display:inline-block;padding:0.2rem 0.55rem;border-radius:0.35rem;"
        f"background:{background};color:{color};font-weight:600;font-size:0.9rem;'>{value}</span>",
        unsafe_allow_html=True,
    )


def _status_colors(status: str) -> tuple[str, str]:
    normalized = str(status).lower()
    if normalized in {"processed", "pass"}:
        return "#dcfce7", "#166534"
    if normalized in {"needs_review", "fail"}:
        return "#fee2e2", "#991b1b"
    return "#fef3c7", "#92400e"


def _pipeline_status_label(status: str) -> str:
    labels = {
        "processed": "Processed",
        "processed_with_warnings": "Processed with warnings",
        "needs_review": "Needs review",
    }
    return labels.get(status, status.replace("_", " ").title())


def _component_status(component: Any) -> str:
    if isinstance(component, dict):
        return str(component.get("status", "unknown")).replace("_", " ").title()
    return "Unknown"


def _classification_label(result: dict[str, Any]) -> str:
    return _classification_label_text(
        result.get("type"),
        result.get("confidence"),
        result.get("confidence_source"),
    )


def _classification_label_text(
    document_type: Any,
    confidence: Any = None,
    confidence_source: Any = None,
) -> str:
    label = _document_type_label(document_type)
    if confidence_source == "heuristic":
        return f"{label} - heuristic match"
    if confidence is not None:
        try:
            return f"{label} - {float(confidence) * 100:.1f}% confident"
        except (TypeError, ValueError):
            return label
    return label


def _document_type_label(document_type: Any) -> str:
    labels = {
        "invoice": "Invoice",
        "receipt": "Receipt",
        "purchase_order": "Purchase Order",
    }
    return labels.get(str(document_type), str(document_type or "Unknown").replace("_", " ").title())


def _fields_from_result(result: dict[str, Any]) -> dict[str, Any]:
    fields = result.get("fields")
    if isinstance(fields, dict):
        return fields

    metadata = result.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("fields"), dict):
        return metadata["fields"]
    return {}


def _display_value(value: Any) -> str:
    return str(value) if value not in (None, "") else "Not found"


def _format_score(score: Any) -> str:
    try:
        return f"{float(score):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_bytes(size: int | None) -> str:
    if size is None:
        return "Unknown"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _highlight_query(text: str, query: str) -> str:
    highlighted = text
    terms = sorted(
        {term.strip(".,;:!?()[]{}").lower() for term in query.split() if len(term.strip()) >= 3},
        key=len,
        reverse=True,
    )
    for term in terms:
        highlighted = re.sub(
            rf"\b({re.escape(term)})\b",
            r"**\1**",
            highlighted,
            flags=re.IGNORECASE,
        )
    return highlighted


if __name__ == "__main__":
    main()
