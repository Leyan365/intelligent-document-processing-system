"""Streamlit dashboard shell for the local IDP pipeline."""

from __future__ import annotations

import base64
import html
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from streamlit_cookies_controller import CookieController
except ImportError:
    CookieController = None
PROJECT_ROOT = SRC_DIR.parent

from idp_system.auth import authenticate_user, initialize_auth_db, register_user
from idp_system.database.document_repository import (
    compute_file_hash,
    get_document_by_id_for_user,
    get_document_by_user_and_hash,
    initialize_document_tables,
    list_documents_for_user,
    save_processed_document,
)
from idp_system.database.session_repository import (
    create_auth_session,
    delete_expired_sessions,
    get_user_for_session_token,
    revoke_auth_session,
    touch_auth_session,
)
from idp_system.system import IDPSystem

SUPPORTED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg"]
UPLOAD_STORAGE_ROOT = Path("data/app/uploads")
PREVIEW_CHARS = 1000
AUTH_COOKIE_NAME = "idp_auth_session"
SHORT_SESSION_HOURS = 12
REMEMBERED_SESSION_DAYS = 7
PAGE_SLUGS = {
    "Upload & Process": "upload",
    "Search": "search",
    "Document History": "history",
}
SLUG_PAGES = {slug: page for page, slug in PAGE_SLUGS.items()}


def main() -> None:
    st.set_page_config(page_title="DocAnalyzer — AI Document Processing", page_icon="📋", layout="wide")
    apply_custom_styles()
    initialize_auth_db()
    initialize_document_tables()
    _ensure_session_state()
    _cleanup_expired_auth_sessions_once()
    _restore_authentication_from_cookie()

    if not st.session_state.authenticated:
        render_auth_page()
        return

    _ensure_active_user_state()
    _load_current_document_result()
    page = _render_authenticated_sidebar()

    if page == "Upload & Process":
        render_upload_page()
    elif page == "Search":
        render_search_page()
    else:
        render_history_page()


def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
        /* ─── Design tokens ─────────────────────────────────────────── */
        :root {
            --c-bg:          #f0f6ff;
            --c-surface:     #ffffff;
            --c-surface-2:   #f5f9ff;
            --c-border:      #dbeafe;
            --c-border-soft: #e8f1fd;
            --c-text:        #0f172a;
            --c-text-muted:  #4b6282;
            --c-primary:     #2563eb;
            --c-primary-dk:  #1d4ed8;
            --c-primary-lt:  #eff6ff;
            --c-primary-mid: #bfdbfe;
            --c-shadow:      0 2px 12px rgba(37, 99, 235, 0.08);
            --c-shadow-lg:   0 8px 32px rgba(37, 99, 235, 0.12);
            --r-card:        1rem;
            --r-btn:         0.6rem;
            --r-input:       0.55rem;
        }

        /* ─── App shell ─────────────────────────────────────────────── */
        .stApp {
            background: var(--c-bg);
            color: var(--c-text);
        }
        .block-container {
            padding-top: 2.25rem;
            padding-bottom: 4rem;
            max-width: 1200px;
        }

        /* ─── Typography ─────────────────────────────────────────────── */
        h1, h2, h3, h4 {
            color: var(--c-text);
            letter-spacing: -0.01em;
            font-weight: 700;
        }
        [data-testid="stCaptionContainer"] { color: var(--c-text-muted); font-size: 0.82rem; }

        /* ─── Sidebar ────────────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: var(--c-surface);
            border-right: 1px solid var(--c-border);
        }
        [data-testid="stSidebar"] * { color: var(--c-text); }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--c-text-muted); }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 0.5rem;
            padding: 0.4rem 0.6rem;
            transition: background 0.15s;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: var(--c-primary-lt);
        }

        /* ─── Cards / bordered containers ───────────────────────────── */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--c-border-soft);
            border-radius: var(--r-card);
            background: var(--c-surface);
            box-shadow: var(--c-shadow);
            padding: 0.25rem 0.25rem;
        }

        /* ─── Native Streamlit metrics ───────────────────────────────── */
        [data-testid="stMetric"] {
            background: var(--c-primary-lt);
            border: 1px solid var(--c-primary-mid);
            border-radius: 0.8rem;
            padding: 1rem 1.1rem;
        }
        [data-testid="stMetricLabel"] p {
            color: var(--c-primary);
            font-weight: 700;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        [data-testid="stMetricValue"] {
            color: var(--c-text);
            font-weight: 800;
            font-size: 1.35rem;
        }

        /* ─── Buttons ────────────────────────────────────────────────── */
        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: var(--r-btn);
            font-weight: 600;
            font-size: 0.9rem;
            border: 1px solid var(--c-border);
            background: var(--c-surface);
            color: var(--c-text);
            transition: all 0.15s;
        }
        div[data-testid="stButton"] > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            border-color: var(--c-primary);
            color: var(--c-primary);
            background: var(--c-primary-lt);
        }
        div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
            background: var(--c-primary);
            color: #ffffff;
            border-color: var(--c-primary);
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover,
        div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
            background: var(--c-primary-dk);
            border-color: var(--c-primary-dk);
            color: #ffffff;
        }

        /* ─── Inputs / textarea ──────────────────────────────────────── */
        input, textarea, [data-baseweb="input"], [data-baseweb="textarea"] {
            border-radius: var(--r-input) !important;
            border-color: var(--c-border) !important;
            background: var(--c-surface) !important;
            color: var(--c-text) !important;
            font-size: 0.95rem !important;
        }
        input:focus, textarea:focus {
            border-color: var(--c-primary) !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
        }
        /* Disabled textarea: keep text legible */
        textarea:disabled,
        [data-baseweb="textarea"] textarea:disabled {
            color: var(--c-text) !important;
            -webkit-text-fill-color: var(--c-text) !important;
            opacity: 1 !important;
            background: var(--c-surface-2) !important;
        }

        /* ─── File uploader ──────────────────────────────────────────── */
        [data-testid="stFileUploader"] section {
            background: var(--c-primary-lt);
            border: 2px dashed var(--c-primary-mid);
            border-radius: var(--r-card);
            padding: 1.5rem;
        }
        [data-testid="stFileUploader"] section:hover {
            border-color: var(--c-primary);
            background: #e0edff;
        }

        /* ─── DataFrame ──────────────────────────────────────────────── */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--c-border);
            border-radius: var(--r-card);
            overflow: hidden;
            box-shadow: var(--c-shadow);
        }

        /* ─── Tabs ───────────────────────────────────────────────────── */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            background: var(--c-surface-2);
            border-radius: 0.6rem;
            padding: 0.25rem;
            gap: 0.25rem;
            border-bottom: none;
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            border-radius: 0.45rem;
            font-weight: 600;
            font-size: 0.9rem;
            color: var(--c-text-muted);
        }
        [data-testid="stTabs"] [aria-selected="true"] {
            background: var(--c-surface) !important;
            color: var(--c-primary) !important;
            box-shadow: var(--c-shadow);
        }

        /* ─── Slider ─────────────────────────────────────────────────── */
        [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
            background: var(--c-primary);
        }

        /* ━━━ Custom component classes ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

        /* Hero / auth shell */
        .idp-auth-shell { max-width: 900px; margin: 0 auto; padding: 3rem 0 1.5rem; }
        .idp-hero { text-align: center; margin-bottom: 2rem; }
        .idp-eyebrow {
            display: inline-flex; align-items: center; gap: 0.4rem;
            padding: 0.3rem 0.85rem; border-radius: 999px;
            background: var(--c-primary-lt); color: var(--c-primary);
            border: 1px solid var(--c-primary-mid);
            font-size: 0.75rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.06em;
            margin-bottom: 0.9rem;
        }
        .idp-hero h1 {
            font-size: clamp(2rem, 4.5vw, 3rem);
            line-height: 1.1; color: var(--c-text);
            margin: 0.5rem 0 0.75rem;
        }
        .idp-hero p {
            color: var(--c-text-muted); font-size: 1.05rem;
            line-height: 1.7; max-width: 640px; margin: 0 auto;
        }

        /* Feature cards on auth page */
        .idp-feature-grid {
            display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem; margin: 1.75rem 0 0.5rem;
        }
        .idp-feature-card {
            background: var(--c-surface); border: 1px solid var(--c-border-soft);
            border-radius: var(--r-card); padding: 1.25rem 1.1rem;
            box-shadow: var(--c-shadow);
        }
        .idp-feature-card .icon {
            font-size: 1.4rem; margin-bottom: 0.6rem; display: block;
        }
        .idp-feature-card strong {
            display: block; color: var(--c-text); font-size: 0.95rem;
            font-weight: 700; margin-bottom: 0.35rem;
        }
        .idp-feature-card span {
            color: var(--c-text-muted); font-size: 0.875rem; line-height: 1.55;
        }

        /* Page header */
        .idp-page-header { margin-bottom: 1.75rem; padding-bottom: 1.25rem; border-bottom: 1px solid var(--c-border-soft); }
        .idp-page-header h1 {
            font-size: clamp(1.6rem, 3vw, 2.1rem);
            margin: 0.5rem 0 0.4rem; color: var(--c-text);
        }
        .idp-page-header p {
            color: var(--c-text-muted); font-size: 0.95rem;
            line-height: 1.6; margin: 0;
        }

        /* Metric grid */
        .idp-metric-grid {
            display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem; margin: 1.15rem 0;
        }
        .idp-metric-grid.wide .idp-metric-card { min-height: 7.2rem; }
        .idp-metric-card {
            background: var(--c-surface); border: 1px solid var(--c-border-soft);
            border-radius: var(--r-card); padding: 1.25rem 1.35rem;
            box-shadow: var(--c-shadow); min-width: 0;
        }
        .idp-metric-card strong {
            display: block; color: var(--c-primary);
            font-size: 0.74rem; font-weight: 800;
            text-transform: uppercase; letter-spacing: 0.05em;
            margin-bottom: 0.45rem;
        }
        .idp-metric-card .value {
            display: block; color: var(--c-text);
            font-size: 1.7rem; font-weight: 850; line-height: 1.2;
            margin-bottom: 0.25rem; overflow-wrap: anywhere;
        }
        .idp-metric-card span:last-child { color: var(--c-text-muted); font-size: 0.86rem; line-height: 1.4; }

        .idp-status-grid {
            display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem; margin-top: 1rem;
        }
        .idp-status-card {
            background: var(--c-surface-2);
            border: 1px solid var(--c-border-soft);
            border-radius: var(--r-card);
            padding: 0.95rem 1rem;
        }
        .idp-status-card strong {
            display: block;
            color: var(--c-text-muted);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
        }
        .idp-pdf-preview {
            border: 1px solid var(--c-border-soft);
            border-radius: var(--r-card);
            width: 100%;
            background: var(--c-surface-2);
        }
        .idp-preview-meta {
            display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem;
            margin: 0.25rem 0 0.85rem;
        }
        .idp-file-name {
            color: var(--c-text); font-size: 0.9rem; font-weight: 700;
            overflow-wrap: anywhere;
        }

        /* Empty state */
        .idp-empty-state {
            text-align: center; padding: 2.5rem 1.5rem;
            background: var(--c-surface); border: 1px dashed var(--c-primary-mid);
            border-radius: var(--r-card);
        }
        .idp-empty-state .empty-icon { font-size: 2rem; margin-bottom: 0.75rem; display: block; }
        .idp-empty-state h3 { color: var(--c-text); margin: 0 0 0.4rem; font-size: 1.05rem; font-weight: 700; }
        .idp-empty-state p { color: var(--c-text-muted); font-size: 0.9rem; line-height: 1.55; margin: 0; }

        /* Info card */
        .idp-info-card {
            background: var(--c-primary-lt); border: 1px solid var(--c-primary-mid);
            border-radius: var(--r-card); padding: 1rem 1.1rem;
        }
        .idp-info-card h3 { color: var(--c-primary-dk); margin: 0 0 0.3rem; font-size: 0.95rem; font-weight: 700; }
        .idp-info-card p { color: var(--c-primary-dk); font-size: 0.88rem; line-height: 1.55; margin: 0; opacity: 0.85; }

        /* Auth note */
        .idp-note {
            background: var(--c-surface-2); border: 1px solid var(--c-border);
            border-radius: 0.7rem; padding: 0.75rem 1rem;
            color: var(--c-text-muted); font-size: 0.82rem; line-height: 1.5;
            margin-top: 0.75rem;
        }

        /* Sidebar brand */
        .idp-sidebar-brand { padding: 0.5rem 0 1rem; }
        .idp-sidebar-brand .mark {
            display: inline-flex; width: 2.4rem; height: 2.4rem;
            align-items: center; justify-content: center;
            border-radius: 0.65rem;
            background: var(--c-primary); color: #fff;
            font-weight: 900; font-size: 0.85rem;
            margin-bottom: 0.65rem;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
        }
        .idp-sidebar-brand h2 { color: var(--c-text); margin: 0; font-size: 1.15rem; font-weight: 800; }
        .idp-sidebar-brand p { margin: 0.2rem 0 0; font-size: 0.8rem; color: var(--c-text-muted); }

        /* Sidebar user card */
        .idp-sidebar-card {
            background: var(--c-primary-lt); border: 1px solid var(--c-primary-mid);
            border-radius: 0.75rem; padding: 0.85rem 1rem; margin: 0.75rem 0 1rem;
        }
        .idp-sidebar-card span {
            display: block; color: var(--c-primary);
            font-size: 0.7rem; text-transform: uppercase;
            letter-spacing: 0.06em; font-weight: 700; margin-bottom: 0.2rem;
        }
        .idp-sidebar-card strong { color: var(--c-text); font-size: 0.95rem; font-weight: 700; }

        /* Badges */
        .idp-badge {
            display: inline-flex; align-items: center;
            padding: 0.28rem 0.75rem; border-radius: 999px;
            font-weight: 700; font-size: 0.8rem; line-height: 1.3;
            border: 1px solid transparent; margin: 0.1rem 0;
        }
        .idp-badge-neutral { background: var(--c-primary-lt); color: var(--c-primary-dk); border-color: var(--c-primary-mid); }
        .idp-badge-success { background: #ecfdf5; color: #065f46; border-color: #a7f3d0; }
        .idp-badge-warning { background: #fffbeb; color: #78350f; border-color: #fde68a; }
        .idp-badge-danger  { background: #fef2f2; color: #991b1b; border-color: #fecaca; }

        /* Pipeline stage tracker */
        .idp-stage-list {
            display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.6rem; padding: 0.85rem; margin: 0.9rem 0;
            background: var(--c-surface); border: 1px solid var(--c-border-soft);
            border-radius: var(--r-card);
        }
        .idp-stage {
            border-radius: 0.55rem; padding: 0.75rem 0.85rem;
            border: 1px solid var(--c-border);
            background: var(--c-surface-2);
            color: var(--c-text-muted); font-size: 0.83rem; font-weight: 600;
        }
        .idp-stage.done   { background: #ecfdf5; color: #065f46; border-color: #a7f3d0; }
        .idp-stage.active { background: var(--c-primary-lt); color: var(--c-primary-dk); border-color: var(--c-primary-mid); }
        .idp-stage.pending { opacity: 0.55; }

        /* Warnings list */
        .idp-warning-list { margin: 0.65rem 0 0; padding-left: 1.1rem; color: #78350f; }
        .idp-warning-list li { margin-bottom: 0.35rem; font-size: 0.88rem; line-height: 1.5; }

        /* Section divider label */
        .idp-section-label {
            font-size: 0.72rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.07em;
            color: var(--c-primary); margin: 1.5rem 0 0.6rem;
        }

        /* ─── Responsive ─────────────────────────────────────────────── */
        @media (max-width: 900px) {
            .idp-feature-grid, .idp-metric-grid, .idp-stage-list { grid-template-columns: 1fr; }
            .block-container { padding-top: 1.25rem; }
            .idp-hero h1 { font-size: 1.8rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_session_state() -> None:
    if "system" not in st.session_state:
        st.session_state.system = None
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
    if "current_document_id" not in st.session_state:
        st.session_state.current_document_id = _document_id_from_query()
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "active_page" not in st.session_state:
        st.session_state.active_page = _page_from_query()
    if "auth_session_token" not in st.session_state:
        st.session_state.auth_session_token = None
    if "auth_restore_attempted" not in st.session_state:
        st.session_state.auth_restore_attempted = False
    if "expired_sessions_cleaned" not in st.session_state:
        st.session_state.expired_sessions_cleaned = False


def _cleanup_expired_auth_sessions_once() -> None:
    if st.session_state.expired_sessions_cleaned:
        return
    delete_expired_sessions()
    st.session_state.expired_sessions_cleaned = True


def _restore_authentication_from_cookie() -> None:
    if st.session_state.authenticated or st.session_state.auth_restore_attempted:
        return

    st.session_state.auth_restore_attempted = True
    token = _read_auth_cookie()
    if not token:
        return

    user = get_user_for_session_token(token)
    if user is None:
        _remove_auth_cookie()
        return

    user_id = int(user["user_id"])
    st.session_state.authenticated = True
    st.session_state.user_id = user_id
    st.session_state.username = user["username"]
    st.session_state.auth_session_token = token
    st.session_state.active_user_id = user_id
    touch_auth_session(token)


def _read_auth_cookie() -> str | None:
    try:
        token = st.context.cookies.get(AUTH_COOKIE_NAME)
    except (AttributeError, KeyError, RuntimeError):
        return None
    return str(token) if token else None


def _set_auth_cookie(token: str, expires_at: datetime) -> bool:
    if CookieController is None:
        st.error(
            "Persistent login requires streamlit-cookies-controller. "
            "Install the documented dependency and restart the app."
        )
        return False

    try:
        if "idp_auth_cookie_controller" not in st.session_state:
            st.session_state.idp_auth_cookie_controller = {}
        controller = CookieController(key="idp_auth_cookie_controller")
        controller.set(
            AUTH_COOKIE_NAME,
            token,
            path="/",
            expires=expires_at,
            secure=_request_uses_https(),
            same_site="lax",
        )
        st.session_state.pop("idp_auth_cookie_set_ack", None)
        CookieController(key="idp_auth_cookie_set_ack")
    except Exception:
        st.error("The browser session cookie could not be set.")
        return False
    return True


def _remove_auth_cookie() -> None:
    if CookieController is None:
        return

    cookie_cache = st.session_state.get("idp_auth_cookie_controller")
    if not isinstance(cookie_cache, dict) or AUTH_COOKIE_NAME not in cookie_cache:
        st.session_state.idp_auth_cookie_controller = {AUTH_COOKIE_NAME: ""}

    try:
        controller = CookieController(key="idp_auth_cookie_controller")
        controller.remove(
            AUTH_COOKIE_NAME,
            path="/",
            secure=_request_uses_https(),
            same_site="lax",
        )
    except Exception:
        pass

    st.session_state.pop("idp_auth_cookie_remove_ack", None)
    CookieController(key="idp_auth_cookie_remove_ack")


def _request_uses_https() -> bool:
    try:
        forwarded_proto = st.context.headers.get("X-Forwarded-Proto", "")
        if str(forwarded_proto).split(",", maxsplit=1)[0].strip().lower() == "https":
            return True
        return str(st.context.url).lower().startswith("https://")
    except (AttributeError, RuntimeError):
        return False


def _page_from_query() -> str:
    try:
        slug = str(st.query_params.get("page", "upload"))
    except (AttributeError, RuntimeError):
        slug = "upload"
    return SLUG_PAGES.get(slug, "Upload & Process")


def _document_id_from_query() -> int | None:
    try:
        value = st.query_params.get("document_id")
        return int(value) if value not in (None, "") else None
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None


def _sync_safe_query_params(page: str | None = None) -> None:
    if page in PAGE_SLUGS:
        st.query_params["page"] = PAGE_SLUGS[page]
    document_id = st.session_state.get("current_document_id")
    if document_id is None:
        st.query_params.pop("document_id", None)
    else:
        st.query_params["document_id"] = str(document_id)


def _clear_safe_query_params() -> None:
    st.query_params.clear()

def _ensure_active_user_state() -> None:
    user_id = st.session_state.user_id
    if user_id is not None and st.session_state.active_user_id != user_id:
        _reset_document_session_state(user_id, clear_navigation=True)
    if user_id is not None and st.session_state.system is None:
        st.session_state.system = IDPSystem()


def _reset_document_session_state(
    user_id: int | None = None,
    *,
    clear_navigation: bool = False,
) -> None:
    st.session_state.system = None
    st.session_state.processed_history = []
    st.session_state.current_document_id = None
    st.session_state.current_result = None
    st.session_state.active_user_id = user_id
    st.session_state.search_index_user_id = None
    st.session_state.search_index_document_count = None
    for key in list(st.session_state):
        if str(key).startswith(("preview_mode_", "image_preview_mode_")):
            del st.session_state[key]
    if clear_navigation:
        st.session_state.active_page = "Upload & Process"
        _clear_safe_query_params()


def render_auth_page() -> None:
    hero_html = (
        '<div class="idp-auth-shell">'
        '<div class="idp-hero">'
        '<span class="idp-eyebrow">AI Document Analyzer</span>'
        '<h1>Intelligent Document<br>Processing System</h1>'
        '<p>Upload invoices, receipts, and purchase orders. Get instant extraction, validation, and semantic search, all running locally.</p>'
        '</div>'
        '<div class="idp-feature-grid">'
        '<div class="idp-feature-card"><span class="icon">Extract</span><strong>Automated Extraction</strong><span>OCR, classification, and field extraction in one seamless local pipeline.</span></div>'
        '<div class="idp-feature-card"><span class="icon">Private</span><strong>Private History</strong><span>Every document stays scoped to your account; no data leaves your machine.</span></div>'
        '<div class="idp-feature-card"><span class="icon">Search</span><strong>Semantic Search</strong><span>Find documents by meaning, supplier, amount, or document type instantly.</span></div>'
        '</div>'
        '</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)

    center = st.columns([1, 1.5, 1])
    with center[1]:
        with st.container(border=True):
            login_tab, register_tab = st.tabs(["Sign in", "Create account"])
            with login_tab:
                render_login_form()
            with register_tab:
                render_register_form()
            note_html = '<div class="idp-note">Local prototype: authentication is for demo purposes only and does not include enterprise controls such as MFA or RBAC.</div>'
            st.markdown(note_html, unsafe_allow_html=True)


def render_login_form() -> None:
    st.markdown("#### Welcome back 👋")
    with st.form("login_form"):
        username_or_email = st.text_input("Username or email")
        password = st.text_input("Password", type="password")
        keep_signed_in = st.checkbox(
            "Keep me signed in for 7 days",
            value=False,
            help="When unchecked, the browser session expires after 12 hours.",
        )
        submitted = st.form_submit_button("Login", type="primary", width="stretch")

    if not submitted:
        return

    result = authenticate_user(username_or_email, password)
    if not result.success or result.user is None:
        st.error(result.error or "Login failed.")
        return

    user_id = int(result.user["user_id"])
    now = datetime.now(timezone.utc)
    expires_at = (
        now + timedelta(days=REMEMBERED_SESSION_DAYS)
        if keep_signed_in
        else now + timedelta(hours=SHORT_SESSION_HOURS)
    )
    token = create_auth_session(user_id, expires_at)
    if not _set_auth_cookie(token, expires_at):
        revoke_auth_session(token)
        return

    _reset_document_session_state(user_id, clear_navigation=True)
    st.session_state.authenticated = True
    st.session_state.user_id = user_id
    st.session_state.username = result.user["username"]
    st.session_state.auth_session_token = token
    st.session_state.auth_restore_attempted = True
    st.success("Login successful.")
    st.stop()


def render_register_form() -> None:
    st.markdown("#### Create your account")
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email (optional)")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Register", type="primary", width="stretch")

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


def _render_authenticated_sidebar() -> str:
    brand_html = (
        '<div class="idp-sidebar-brand">'
        '<div class="mark">IDP</div>'
        '<h2>DocAnalyzer</h2>'
        '<p>AI-powered document intelligence</p>'
        '</div>'
    )
    st.sidebar.markdown(brand_html, unsafe_allow_html=True)
    user_html = (
        '<div class="idp-sidebar-card">'
        '<span>Signed in as</span>'
        f'<strong>{_html_escape(st.session_state.username)}</strong>'
        '</div>'
    )
    st.sidebar.markdown(user_html, unsafe_allow_html=True)
    st.sidebar.caption("Navigation")
    page = st.sidebar.radio(
        "Navigation",
        ["Upload & Process", "Search", "Document History"],
        label_visibility="collapsed",
        key="active_page",
    )
    _sync_safe_query_params(page)
    st.sidebar.divider()
    st.sidebar.caption("Local demo ? per-user document isolation")
    if st.sidebar.button("Sign out", width="stretch"):
        _logout()
    return page


def _logout() -> None:
    token = st.session_state.get("auth_session_token") or _read_auth_cookie()
    revoke_auth_session(token)
    _remove_auth_cookie()
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.auth_session_token = None
    st.session_state.auth_restore_attempted = True
    _reset_document_session_state(None, clear_navigation=True)
    st.rerun()


def render_upload_page() -> None:
    _page_header(
        "Upload & Process",
        "Drop in a PDF or image and the pipeline handles everything — text extraction, classification, field parsing, and search indexing.",
    )

    with st.container(border=True):
        st.markdown("### 📄 Choose a document")
        st.caption(
            "Supports digital PDFs, scanned PDFs, and document images (PNG, JPG, JPEG). Processing runs entirely on your local machine."
        )
        uploaded_file = st.file_uploader(
            "Choose a PDF or image file",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        st.caption("Accepted formats: PDF · PNG · JPG · JPEG")

    current_result = st.session_state.get("current_result")
    if uploaded_file is None:
        if current_result is not None:
            render_result(current_result)
            return
        _empty_state(
            "No document selected",
            "Upload a PDF or image above to begin — the pipeline will extract, classify, and validate it automatically.",
        )
        return

    with st.container(border=True):
        st.markdown("### 📎 Selected file")
        col_name, col_type, col_size = st.columns(3)
        col_name.metric("Name", uploaded_file.name)
        col_type.metric("Type", uploaded_file.type or "Unknown")
        col_size.metric("Size", _format_bytes(uploaded_file.size))

    if st.button("⚡ Run Processing Pipeline", type="primary", width="stretch"):
        user_id = _current_user_id()
        file_bytes = uploaded_file.getvalue()
        file_hash = compute_file_hash(file_bytes)
        duplicate = get_document_by_user_and_hash(user_id, file_hash)

        if duplicate is not None:
            result = _result_from_document_record(duplicate)
            _remember_result(result)
            _info_card(
                "Already processed",
                "This exact file was found in your document history — the saved result has been loaded below.",
            )
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
            # Clear the stage tracker and progress bar so they don't persist below results
            stages.empty()
            progress.empty()

            st.toast("Document processed and saved successfully")
        except Exception as exc:
            st.error("Document processing failed.")
            st.exception(exc)

    current_result = st.session_state.get("current_result")
    if current_result is not None:
        render_result(current_result)


def render_result(result: dict[str, Any]) -> None:
    st.markdown("## Processing result")
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    score = validation.get("validation_score") if validation else None
    warnings = validation.get("total_warnings") if validation else None

    _metric_grid(
        [
            ("Document type", _classification_label(result), "Detected class"),
            ("Validation score", _display_value(score), "Overall confidence"),
            ("Warnings", _display_value(warnings), "Items requiring attention"),
        ],
        wide=True,
    )
    st.divider()

    preview_col, text_col = st.columns([1.1, 1], gap="large")
    with preview_col:
        render_document_preview(
            result.get("stored_path"),
            result.get("source_filename"),
            document_id=result.get("persistent_document_id"),
        )
    with text_col:
        with st.container(border=True):
            st.markdown("### Extracted text preview")
            text = str(result.get("text", ""))
            preview = _snippet(text, PREVIEW_CHARS)
            st.text_area(
                "Extracted text preview",
                value=preview,
                height=340,
                disabled=True,
                label_visibility="collapsed",
            )
            if len(text) > PREVIEW_CHARS:
                st.caption(f"Showing first {PREVIEW_CHARS:,} characters of extracted text.")

    detail_col, validation_col = st.columns([1, 1], gap="large")
    with detail_col:
        with st.container(border=True):
            st.markdown("### Classification")
            _badge(_classification_label(result))
            source_filename = result.get("source_filename")
            if source_filename:
                st.caption(f"File: {_display_value(source_filename)}")

        render_extracted_fields(result)

    with validation_col:
        render_validation_section(result)


def render_validation_section(result: dict[str, Any]) -> None:
    validation = result.get("validation")
    if not isinstance(validation, dict):
        return

    with st.container(border=True):
        st.markdown("### Validation & confidence")
        pipeline_status = str(validation.get("pipeline_status", "processed"))
        _status_badge(_pipeline_status_label(pipeline_status), pipeline_status)

        score = validation.get("validation_score")
        total_warnings = validation.get("total_warnings", 0)
        critical_count = validation.get("critical_warning_count", 0)
        _metric_grid(
            [
                ("Validation score", _display_value(score), "Combined quality signal"),
                ("Total warnings", _display_value(total_warnings), "Review indicators found"),
                ("Critical warnings", _display_value(critical_count), "High-priority issues"),
            ],
            wide=True,
        )

        _status_card_grid(
            [
                ("OCR quality", _component_status(validation.get("ocr_quality")), _component_status_key(validation.get("ocr_quality"))),
                ("Classification", _component_status(validation.get("classification")), _component_status_key(validation.get("classification"))),
                ("Fields", _component_status(validation.get("fields")), _component_status_key(validation.get("fields"))),
            ]
        )

        warnings = validation.get("warnings")
        if isinstance(warnings, list) and warnings:
            shown = [str(warning) for warning in warnings[:4]]
            warning_items = "".join(f"<li>{_html_escape(warning)}</li>" for warning in shown)
            extra = len(warnings) - len(shown)
            if extra > 0:
                warning_items += f"<li>{_html_escape(f'+ {extra} more warnings')}</li>"
            warning_html = f'<div class="idp-info-card"><h3>Items to review</h3><ul class="idp-warning-list">{warning_items}</ul></div>'
            st.markdown(warning_html, unsafe_allow_html=True)


def render_search_page() -> None:
    _page_header(
        "Search Documents",
        "Search across all processed documents using natural language — by supplier, amount, date, or document content.",
    )
    with st.spinner("Building search index..."):
        _ensure_search_index_for_current_user()

    search_service = st.session_state.system.search_service
    if not search_service.semantic_search_available:
        st.info(
            "Semantic model is unavailable locally. Showing fast exact and keyword matches instead. "
            "Download the MiniLM model once while online to re-enable semantic ranking."
        )

    with st.container(border=True):
        st.markdown("### 🔎 Search your documents")
        query = st.text_input(
            "Search documents",
            placeholder="e.g. invoice from Lalan Rubbers, purchase order over 5000, receipt for office supplies",
            label_visibility="collapsed",
        )
        st.caption("Tip: search by meaning, not just keywords — try supplier names, amounts, or document descriptions.")
        k = st.slider("Number of results to show", min_value=1, max_value=10, value=5)

    if not query:
        _empty_state(
            "Ready to search",
            "Type a query above to find documents by supplier, amount, date, or content.",
        )
        return

    try:
        results = st.session_state.system.search(query, k=k)
    except ValueError as exc:
        st.warning(str(exc))
        return
    except Exception as exc:
        st.error("Search failed.")
        st.exception(exc)
        return

    if not results:
        _empty_state(
            "No matching documents found",
            "Try a different supplier, amount, document type, or phrase from the document.",
        )
        return

    st.markdown("## Search results")
    for rank, result in enumerate(results, start=1):
        with st.container(border=True):
            fields = _fields_from_result(result)
            metadata = result.get("metadata") or {}
            doc_type = result.get("type") or metadata.get("type") or "unknown"
            confidence = result.get("confidence", metadata.get("confidence"))
            confidence_source = result.get("confidence_source", metadata.get("confidence_source"))
            filename = result.get("filename") or metadata.get("filename") or metadata.get("source")

            header_left, header_right = st.columns([3, 1])
            with header_left:
                st.markdown(f"### #{rank} — {_display_value(filename) if filename else 'Document'}")
                _badge(_classification_label_text(doc_type, confidence, confidence_source))
                if filename:
                    st.caption(f"📁 {_display_value(filename)}")
            with header_right:
                st.metric("Match score", _format_score(result.get("score")))

            st.divider()
            st.markdown("**Key information**")
            info_cols = st.columns(4)
            info_cols[0].markdown(f"**Supplier**  \n{_display_value(fields.get('supplier'))}")
            info_cols[1].markdown(f"**Date**  \n{_display_value(fields.get('date'))}")
            info_cols[2].markdown(f"**Amount**  \n{_display_value(fields.get('amount'))}")
            info_cols[3].markdown(f"**Invoice / Order No.**  \n{_display_value(fields.get('invoice_number'))}")
            st.markdown("**Matched content preview**")
            st.markdown(_highlight_query(_snippet(str(result.get("text", "")), 280), query))


def render_history_page() -> None:
    _page_header(
        "Document History",
        "All documents processed under your account, stored locally in the SQLite database.",
    )
    records = list_documents_for_user(_current_user_id())

    if not records:
        _empty_state(
            "No documents yet",
            "Upload and process your first document — it will appear here once complete.",
        )
        return

    rows = []
    for record in records:
        result = _result_from_document_record(record)
        fields = result.get("fields") or {}
        validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
        rows.append(
            {
                "filename": record.get("original_filename"),
                "document_type": _document_type_label(result.get("type")),
                "created_at": record.get("created_at"),
                "validation_status": _pipeline_status_label(str(validation.get("pipeline_status", "processed"))),
                "supplier": fields.get("supplier"),
                "date": fields.get("date"),
                "amount": fields.get("amount"),
            }
        )

    # Full sortable table in expander — cards below are the primary view
    with st.expander(f"📋 View all {len(rows)} documents as table", expanded=False):
        st.dataframe(rows, width="stretch", hide_index=True)

    st.markdown("## Recent documents")
    for record in records[:5]:
        result = _result_from_document_record(record)
        fields = result.get("fields") or {}
        validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
        with st.container(border=True):
            top_left, top_right = st.columns([3, 1])
            with top_left:
                st.markdown(f"### 📄 {_display_value(record.get('original_filename'))}")
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
            st.divider()
            cols = st.columns(3)
            cols[0].metric("Supplier", _display_value(fields.get("supplier")))
            cols[1].metric("Date", _display_value(fields.get("date")))
            cols[2].metric("Amount", _display_value(fields.get("amount")))
            document_id = int(record["document_id"])
            st.button(
                "Review document",
                key=f"review_document_{document_id}",
                type="secondary",
                width="stretch",
                on_click=_open_document_for_review,
                args=(document_id,),
            )
    st.caption("Showing 5 most recent · expand the table above to see all documents")


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
    result["stored_path"] = record.get("stored_path")
    return result



def _load_current_document_result() -> dict[str, Any] | None:
    document_id = st.session_state.get("current_document_id")
    if document_id is None:
        st.session_state.current_result = None
        return None

    try:
        validated_id = int(document_id)
    except (TypeError, ValueError):
        st.session_state.current_document_id = None
        st.session_state.current_result = None
        _sync_safe_query_params()
        return None

    record = get_document_by_id_for_user(validated_id, _current_user_id())
    if record is None:
        st.session_state.current_document_id = None
        st.session_state.current_result = None
        _sync_safe_query_params()
        return None

    result = _result_from_document_record(record)
    st.session_state.current_document_id = validated_id
    st.session_state.current_result = result
    return result


def _open_document_for_review(document_id: int) -> None:
    record = get_document_by_id_for_user(int(document_id), _current_user_id())
    if record is None:
        return
    _remember_result(_result_from_document_record(record))
    st.session_state.active_page = "Upload & Process"
    _sync_safe_query_params("Upload & Process")


def _remember_result(result: dict[str, Any]) -> None:
    document_id = result.get("persistent_document_id") or result.get("id")
    history = st.session_state.processed_history
    if document_id is not None:
        document_id = int(document_id)
        st.session_state.current_document_id = document_id
        history[:] = [
            item for item in history
            if (item.get("persistent_document_id") or item.get("id")) != document_id
        ]
    history.append(result)
    st.session_state.current_result = result
    _sync_safe_query_params()


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
                "filename": record.get("original_filename"),
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
    stage_names = ["Text Extraction", "Classification", "Information Extraction", "Search Indexing"]
    stage_html = []
    for index, stage_name in enumerate(stage_names):
        if index < active_index:
            state = "done"
        elif index == active_index:
            state = "active"
        else:
            state = "pending"
        stage_html.append(f'<div class="idp-stage {state}">{_html_escape(stage_name)}</div>')
    stage_markup = "".join(stage_html)
    container.markdown(f'<div class="idp-stage-list">{stage_markup}</div>', unsafe_allow_html=True)


def _snippet(text: str, length: int = 500) -> str:
    text = " ".join(text.split())
    return text[:length] + ("..." if len(text) > length else "")


def render_extracted_fields(result: dict[str, Any]) -> None:
    st.markdown("### Extracted fields")
    st.caption("Fields are editable; correct any extraction errors below.")
    field_layout = (
        ("invoice_number", "Invoice / Order No."),
        ("amount", "Amount"),
        ("date", "Date"),
        ("supplier", "Supplier"),
    )
    for row_start in range(0, len(field_layout), 2):
        cols = st.columns(2, gap="medium")
        for col, (field_name, label) in zip(cols, field_layout[row_start:row_start + 2]):
            with col:
                _editable_field_card(result, field_name, label)


def _editable_field_card(result: dict[str, Any], field_name: str, label: str) -> None:
    fields = result.setdefault("fields", {})
    document_id = result.get("persistent_document_id") or result.get("id", "document")
    with st.container(border=True):
        st.markdown(f"**{label}**")
        updated_value = st.text_input(
            label,
            value="" if fields.get(field_name) is None else str(fields.get(field_name)),
            key=f"{document_id}_{field_name}",
            label_visibility="collapsed",
            placeholder=f"Enter {label.lower()}...",
        )
        fields[field_name] = updated_value or None


def _badge(value: str) -> None:
    st.markdown(f'<span class="idp-badge idp-badge-neutral">{_html_escape(value)}</span>', unsafe_allow_html=True)


def _status_badge(value: str, status: str) -> None:
    badge_class = _status_badge_class(status)
    st.markdown(f'<span class="idp-badge {badge_class}">{_html_escape(value)}</span>', unsafe_allow_html=True)


def _status_badge_class(status: str) -> str:
    normalized = str(status).lower()
    if normalized in {"processed", "pass"}:
        return "idp-badge-success"
    if normalized in {"needs_review", "fail"}:
        return "idp-badge-danger"
    return "idp-badge-warning"


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


def _component_status_key(component: Any) -> str:
    if isinstance(component, dict):
        return str(component.get("status", "unknown")).lower()
    return "unknown"


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


def render_document_preview(
    file_path: str | Path | None,
    filename: str | None = None,
    *,
    document_id: Any = None,
) -> None:
    with st.container(border=True):
        st.markdown("### Original document preview")
        path = _resolve_preview_path(file_path)
        if path is None:
            st.info("Original file preview is unavailable for this record.")
            return

        try:
            file_bytes = path.read_bytes()
        except OSError:
            st.info("Original file preview is unavailable for this record.")
            return

        display_name = filename or path.name
        suffix = path.suffix.lower()
        mime_type = _preview_mime_type(path)
        preview_key = str(document_id) if document_id is not None else _preview_key(path)
        _preview_file_header(display_name, suffix)
        _download_original_file(file_bytes, display_name, mime_type)

        if suffix == ".pdf":
            preview_mode = st.radio(
                "Preview size",
                ["Normal preview", "Large preview"],
                horizontal=True,
                key=f"preview_mode_{preview_key}",
            )
            st.caption("Use large preview mode to compare the source document with extracted fields.")
            render_pdf_preview(file_bytes, height=1000 if preview_mode == "Large preview" else 650)
        elif suffix in {".png", ".jpg", ".jpeg"}:
            preview_mode = st.radio(
                "Image preview mode",
                ["Fit to page width", "Large preview"],
                horizontal=True,
                key=f"image_preview_mode_{preview_key}",
            )
            render_image_preview(file_bytes, display_name, large=preview_mode == "Large preview")
        else:
            st.info("Original file preview is unavailable for this record.")


def render_pdf_preview(file_bytes: bytes, height: int = 650) -> None:
    encoded = base64.b64encode(file_bytes).decode("ascii")
    iframe = (
        '<iframe class="idp-pdf-preview" '
        f'style="height:{int(height)}px" '
        f'src="data:application/pdf;base64,{encoded}#toolbar=1&navpanes=0"></iframe>'
    )
    st.markdown(iframe, unsafe_allow_html=True)
    st.caption("If the browser cannot preview this PDF, use the download button above.")


def render_image_preview(file_bytes: bytes, filename: str | None = None, large: bool = False) -> None:
    try:
        if large:
            st.image(file_bytes, caption=filename, width=1100)
            st.caption("Large preview renders the image at an expanded width for closer inspection.")
        else:
            st.image(file_bytes, caption=filename, use_container_width=True)
    except Exception:
        st.info("Original file preview is unavailable for this record.")


def _preview_file_header(filename: str, suffix: str) -> None:
    label = _preview_file_type_label(suffix)
    html_value = (
        '<div class="idp-preview-meta">'
        f'<span class="idp-badge idp-badge-neutral">{_html_escape(label)}</span>'
        f'<span class="idp-file-name">{_html_escape(filename)}</span>'
        '</div>'
    )
    st.markdown(html_value, unsafe_allow_html=True)


def _download_original_file(file_bytes: bytes, filename: str, mime_type: str) -> None:
    st.download_button(
        "Download original file",
        data=file_bytes,
        file_name=filename,
        mime=mime_type,
        type="secondary",
        width="stretch",
    )


def _preview_mime_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "application/octet-stream"


def _preview_file_type_label(suffix: str) -> str:
    labels = {
        ".pdf": "PDF",
        ".png": "PNG image",
        ".jpg": "JPEG image",
        ".jpeg": "JPEG image",
    }
    return labels.get(suffix.lower(), "File")


def _resolve_preview_path(file_path: str | Path | None) -> Path | None:
    if file_path in (None, ""):
        return None

    raw_path = Path(str(file_path))
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.append(PROJECT_ROOT / raw_path)
        candidates.append(Path.cwd() / raw_path)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _preview_key(file_path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(file_path.resolve()))


def _page_header(title: str, description: str) -> None:
    html_value = (
        '<div class="idp-page-header">'
        '<span class="idp-eyebrow">IDP Dashboard</span>'
        f'<h1>{_html_escape(title)}</h1>'
        f'<p>{_html_escape(description)}</p>'
        '</div>'
    )
    st.markdown(html_value, unsafe_allow_html=True)


def _empty_state(title: str, description: str) -> None:
    html_value = (
        '<div class="idp-empty-state">'
        '<span class="empty-icon">No records</span>'
        f'<h3>{_html_escape(title)}</h3>'
        f'<p>{_html_escape(description)}</p>'
        '</div>'
    )
    st.markdown(html_value, unsafe_allow_html=True)


def _info_card(title: str, description: str) -> None:
    html_value = (
        '<div class="idp-info-card">'
        f'<h3>{_html_escape(title)}</h3>'
        f'<p>{_html_escape(description)}</p>'
        '</div>'
    )
    st.markdown(html_value, unsafe_allow_html=True)


def _metric_grid(metrics: list[tuple[str, str, str]], wide: bool = False) -> None:
    cards = []
    for label, value, caption in metrics:
        cards.append(
            '<div class="idp-metric-card">'
            f'<strong>{_html_escape(label)}</strong>'
            f'<span class="value">{_html_escape(value)}</span>'
            f'<span>{_html_escape(caption)}</span>'
            '</div>'
        )
    grid_class = 'idp-metric-grid wide' if wide else 'idp-metric-grid'
    st.markdown(f'<div class="{grid_class}">{"".join(cards)}</div>', unsafe_allow_html=True)


def _status_card_grid(statuses: list[tuple[str, str, str]]) -> None:
    cards = []
    for label, value, status in statuses:
        badge_class = _status_badge_class(status)
        cards.append(
            '<div class="idp-status-card">'
            f'<strong>{_html_escape(label)}</strong>'
            f'<span class="idp-badge {badge_class}">{_html_escape(value)}</span>'
            '</div>'
        )
    st.markdown(f'<div class="idp-status-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _html_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
