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
PREVIEW_CHARS = 1800


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
    uploaded_file = st.file_uploader(
        "Choose a PDF or image file",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload a PDF or image document to begin processing.")
        return

    st.caption(f"Selected file: {uploaded_file.name}")
    if st.button("Process Document", type="primary"):
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
            preview = str(result.get("text", ""))[:PREVIEW_CHARS]
            st.text_area("Extracted text", value=preview, height=320, disabled=True)

    with right:
        with st.container(border=True):
            st.markdown("**Extracted Fields**")
            st.metric("Document Type", str(result.get("type") or "unknown"))
            fields = result.get("fields") or {}
            for field_name in ("invoice_number", "date", "amount", "supplier"):
                st.write(f"**{field_name.replace('_', ' ').title()}**")
                st.write(fields.get(field_name) or "Not found")


def render_search_page() -> None:
    st.header("Search")
    query = st.text_input("Search documents", placeholder="Search by meaning, supplier, amount, or document type")
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

    for rank, result in enumerate(results, start=1):
        with st.container(border=True):
            score = result.get("score")
            score_label = f"{float(score):.3f}" if score is not None else "N/A"
            st.markdown(f"**#{rank} - {result.get('id', 'unknown')}**")
            st.caption(f"Score: {score_label}")
            st.write(_snippet(str(result.get("text", ""))))


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
                "type": document.get("type"),
                "supplier": fields.get("supplier"),
                "date": fields.get("date"),
                "amount": fields.get("amount"),
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True)
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


if __name__ == "__main__":
    main()
