"""Loader abstractions for source documents."""

from abc import ABC, abstractmethod
from pathlib import Path

from ..core.exceptions import DocumentLoadError, UnsupportedDocumentTypeError
from ..core.models import Document, DocumentType


class BaseLoader(ABC):
    """Base class for document loaders."""

    supported_types: tuple[DocumentType, ...] = ()

    def supports(self, doc_type: DocumentType) -> bool:
        return doc_type in self.supported_types

    @abstractmethod
    def load(self, source: str | Path) -> Document:
        """Load a document from a source."""


class PlaceholderFileLoader(BaseLoader):
    """Phase 1 placeholder loader for supported local file types."""

    supported_types = (
        DocumentType.PDF,
        DocumentType.IMAGE,
        DocumentType.TXT,
        DocumentType.DOCX,
        DocumentType.MD,
        DocumentType.HTML,
        DocumentType.JSON,
        DocumentType.CSV,
    )

    def load(self, source: str | Path) -> Document:
        source_path = Path(source)
        if not source_path.exists():
            raise DocumentLoadError(f"File not found: {source_path}")

        doc_type = document_type_from_path(source_path)
        return Document.empty_from_path(
            source_path,
            doc_type=doc_type,
        )


class DocumentLoaderRouter:
    """Selects the correct loader for an input source."""

    def __init__(self, loaders: tuple[BaseLoader, ...] | None = None) -> None:
        self.loaders = loaders or (PlaceholderFileLoader(),)

    def load(self, source: str | Path) -> Document:
        doc_type = document_type_from_path(source)
        for loader in self.loaders:
            if loader.supports(doc_type):
                return loader.load(source)
        raise UnsupportedDocumentTypeError(f"Unsupported document type: {doc_type.value}")


def document_type_from_path(source: str | Path) -> DocumentType:
    """Infer document type from a path extension."""
    suffix = Path(source).suffix.lower()
    if suffix == ".pdf":
        return DocumentType.PDF
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        return DocumentType.IMAGE
    try:
        return DocumentType(suffix.lstrip("."))
    except ValueError as exc:
        raise UnsupportedDocumentTypeError(
            f"Unsupported document extension: {suffix or '<none>'}"
        ) from exc
