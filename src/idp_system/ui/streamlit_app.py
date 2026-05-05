"""Streamlit dashboard shell for the local IDP pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from idp_system.system import IDPSystem


SUPPORTED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg"]
TEMP_UPLOAD_DIR = Path("temp_uploads")
PREVIEW_CHARS = 1000


def main() -> None:
    st.set_page_config(
        page_title="Intelligent Document Processing System",
        layout="wide",
    )
    _ensure_session_state()

    st.title("Intelligent Document Processing System")

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
            st.markdown("**No document selected**")
            st.write("Supported formats: PDF, PNG, JPG, and JPEG.")
            st.write("After upload, use **Process Document** to run the local IDP pipeline.")
        return

    with st.container(border=True):
        st.markdown("**Selected File**")
        col_name, col_type, col_size = st.columns(3)
        col_name.metric("Name", uploaded_file.name)
        col_type.metric("Type", uploaded_file.type or "Unknown")
        col_size.metric("Size", _format_bytes(uploaded_file.size))

    if st.button("Process Document", type="primary", width="stretch"):
        temp_path = _save_uploaded_file(uploaded_file)
        progress = st.progress(0)
        stages = st.empty()

        try:
            _render_stages(stages, active_index=0)
            progress.progress(20, text="Text Extraction")

            _render_stages(stages, active_index=1)
            progress.progress(45, text="Classification")

            _render_stages(stages, active_index=2)
            progress.progress(70, text="Information Extraction")

            result = st.session_state.system.process_document(temp_path)

            _render_stages(stages, active_index=3)
            progress.progress(90, text="Search Indexing")

            st.session_state.processed_history.append(result)
            _render_stages(stages, active_index=4)
            progress.progress(100, text="Complete")

            st.success("Document processed successfully.")
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
            _badge(str(result.get("type") or "unknown"))

        st.markdown("**Extracted Fields**")
        field_cols = st.columns(2)
        fields = result.get("fields") or {}
        for index, field_name in enumerate(("invoice_number", "date", "amount", "supplier")):
            with field_cols[index % 2]:
                _field_card(field_name, fields.get(field_name))


def render_search_page() -> None:
    st.header("Search")
    st.info(
        "Semantic search retrieves relevant processed documents and snippets. "
        "It does not generate answers."
    )

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
        st.warning("No matching documents found.")
        return

    st.subheader("Search Results")

    for rank, result in enumerate(results, start=1):
        with st.container(border=True):
            fields = _fields_from_result(result)
            metadata = result.get("metadata") or {}
            doc_type = result.get("type") or metadata.get("type") or "unknown"

            header_left, header_right = st.columns([3, 1])
            with header_left:
                st.markdown(f"**#{rank} - {result.get('id', 'unknown')}**")
                _badge(str(doc_type))
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
                f"**Invoice No.**  \n`{_display_value(fields.get('invoice_number'))}`"
            )

            st.markdown("**Matched Preview**")
            st.write(_snippet(str(result.get("text", "")), 200))


def render_history_page() -> None:
    st.header("Document History")
    history = st.session_state.processed_history

    if not history:
        st.info("No processed documents yet.")
        return

    rows = []
    for document in history:
        fields = document.get("fields") or {}
        rows.append(
            {
                "id": document.get("id"),
                "status": "Processed",
                "type": document.get("type"),
                "supplier": fields.get("supplier"),
                "date": fields.get("date"),
                "amount": fields.get("amount"),
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True)

    st.markdown("**Recent Documents**")
    for document in reversed(history[-5:]):
        fields = document.get("fields") or {}
        with st.container(border=True):
            top_left, top_right = st.columns([3, 1])
            with top_left:
                st.markdown(f"**{document.get('id')}**")
                _badge(str(document.get("type") or "unknown"))
            with top_right:
                st.markdown("`Processed`")
            cols = st.columns(3)
            cols[0].caption(f"Supplier: {_display_value(fields.get('supplier'))}")
            cols[1].caption(f"Date: {_display_value(fields.get('date'))}")
            cols[2].caption(f"Amount: {_display_value(fields.get('amount'))}")
    st.caption("Document detail buttons will be added in a later UI phase.")


def _save_uploaded_file(uploaded_file: Any) -> Path:
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix
    safe_name = f"{uuid4().hex}{suffix}"
    temp_path = TEMP_UPLOAD_DIR / safe_name
    temp_path.write_bytes(uploaded_file.getbuffer())
    return temp_path


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
            marker = "[done]"
        elif index == active_index:
            marker = "[running]"
        else:
            marker = "[pending]"
        lines.append(f"{marker} {stage_name}")
    container.code("\n".join(lines), language="text")


def _snippet(text: str, length: int = 500) -> str:
    text = " ".join(text.split())
    return text[:length] + ("..." if len(text) > length else "")


def _field_card(label: str, value: Any) -> None:
    with st.container(border=True):
        st.caption(label.replace("_", " ").title())
        st.markdown(f"`{_display_value(value)}`")


def _badge(value: str) -> None:
    st.markdown(
        f"<span style='display:inline-block;padding:0.2rem 0.55rem;border-radius:0.35rem;"
        f"background:#eef2ff;color:#1f2937;font-weight:600;font-size:0.9rem;'>{value}</span>",
        unsafe_allow_html=True,
    )


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


if __name__ == "__main__":
    main()
