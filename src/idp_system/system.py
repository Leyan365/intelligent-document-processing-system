"""Top-level orchestrator for the local IDP system."""

from pathlib import Path
from typing import Iterable

from .core.models import Document
from .pipeline.loader import DocumentLoaderRouter


class IDPSystem:
    """Coordinates the document processing pipeline.

    Phase 1 only wires the package structure and loader entry point. OCR,
    classification, extraction, persistence, and search are intentionally left
    as placeholders for later phases.
    """

    def __init__(self, loader: DocumentLoaderRouter | None = None) -> None:
        self.loader = loader or DocumentLoaderRouter()
        self.documents: dict[str, Document] = {}

    def load_document(self, source: str | Path) -> Document:
        """Load one document through the router."""
        document = self.loader.load(source)
        self.documents[document.id] = document
        return document

    def load_documents(self, sources: Iterable[str | Path]) -> list[Document]:
        """Load multiple documents through the router."""
        return [self.load_document(source) for source in sources]

    def process_document(self, source: str | Path) -> Document:
        """Phase 1 placeholder for the future end-to-end pipeline."""
        return self.load_document(source)
