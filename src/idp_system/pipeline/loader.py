"""Loader abstractions and text extraction for source documents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import re

from ..core.exceptions import DocumentLoadError, UnsupportedDocumentTypeError
from ..core.models import Document, DocumentType
from .ocr import OCRService


class BaseLoader(ABC):
    """Base class for document loaders."""

    supported_types: tuple[DocumentType, ...] = ()

    def supports(self, doc_type: DocumentType) -> bool:
        return doc_type in self.supported_types

    @abstractmethod
    def load(self, source: str | Path) -> Document:
        """Load a document from a source."""


@dataclass(slots=True)
class TextExtractionResult:
    """Cleaned extraction output plus basic provenance metadata."""

    text: str
    doc_type: DocumentType
    extraction_method: str
    metadata: dict[str, object] = field(default_factory=dict)


class LocalTextExtractionLoader(BaseLoader):
    """Extract text from local PDF, image, and simple text files."""

    supported_types = (
        DocumentType.PDF,
        DocumentType.IMAGE,
        DocumentType.TXT,
        DocumentType.MD,
        DocumentType.HTML,
        DocumentType.JSON,
        DocumentType.CSV,
    )

    def __init__(self, ocr_service: OCRService | None = None, min_pdf_text_chars: int = 50) -> None:
        self.ocr_service = ocr_service or OCRService()
        self.min_pdf_text_chars = min_pdf_text_chars

    def load(self, source: str | Path) -> Document:
        source_path = Path(source)
        if not source_path.exists():
            raise DocumentLoadError(f"File not found: {source_path}")

        result = extract_text(source_path, self.ocr_service, self.min_pdf_text_chars)
        return Document(
            title=source_path.stem,
            content=result.text,
            source=str(source_path),
            doc_type=result.doc_type,
            metadata=result.metadata,
            extraction_method=result.extraction_method,
        )


class DocumentLoaderRouter:
    """Selects the correct loader for an input source."""

    def __init__(self, loaders: tuple[BaseLoader, ...] | None = None) -> None:
        self.loaders = loaders or (LocalTextExtractionLoader(),)

    def load(self, source: str | Path) -> Document:
        doc_type = document_type_from_path(source)
        for loader in self.loaders:
            if loader.supports(doc_type):
                return loader.load(source)
        raise UnsupportedDocumentTypeError(f"Unsupported document type: {doc_type.value}")


def extract_text(
    source: str | Path,
    ocr_service: OCRService | None = None,
    min_pdf_text_chars: int = 50,
) -> TextExtractionResult:
    """Extract cleaned text from a supported file.

    PDFs are attempted with PyMuPDF first. If direct text is too short, pages are
    rendered locally and passed through PaddleOCR.
    """
    source_path = Path(source)
    if not source_path.exists():
        raise DocumentLoadError(f"File not found: {source_path}")

    doc_type = document_type_from_path(source_path)
    metadata = _file_metadata(source_path)
    ocr = ocr_service or OCRService()

    if doc_type == DocumentType.PDF:
        direct_text, page_count = _extract_pdf_text_with_pymupdf(source_path)
        metadata["page_count"] = page_count
        metadata["direct_text_chars"] = len(direct_text)

        if len(direct_text.strip()) >= min_pdf_text_chars:
            return TextExtractionResult(
                text=clean_text(direct_text),
                doc_type=doc_type,
                extraction_method="pymupdf",
                metadata=metadata,
            )

        ocr_text = _extract_pdf_text_with_ocr(source_path, ocr)
        metadata["ocr_fallback"] = True
        return TextExtractionResult(
            text=clean_text(ocr_text),
            doc_type=doc_type,
            extraction_method="paddleocr_pdf_fallback",
            metadata=metadata,
        )

    if doc_type == DocumentType.IMAGE:
        text = ocr.extract_text(source_path)
        return TextExtractionResult(
            text=clean_text(text),
            doc_type=doc_type,
            extraction_method="paddleocr_image",
            metadata=metadata,
        )

    if doc_type in {DocumentType.TXT, DocumentType.MD, DocumentType.HTML, DocumentType.JSON, DocumentType.CSV}:
        return TextExtractionResult(
            text=clean_text(source_path.read_text(encoding="utf-8")),
            doc_type=doc_type,
            extraction_method="plain_text",
            metadata=metadata,
        )

    raise UnsupportedDocumentTypeError(f"Text extraction is not implemented for: {doc_type.value}")


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


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _extract_pdf_text_with_pymupdf(path: Path) -> tuple[str, int]:
    try:
        import fitz
    except ImportError as exc:
        raise DocumentLoadError(
            "PyMuPDF is required for PDF text extraction. Install package: pymupdf"
        ) from exc

    try:
        pdf = fitz.open(path)
    except Exception as exc:
        raise DocumentLoadError(
            f"Could not open PDF with PyMuPDF: {path}. Original error: {exc}"
        ) from exc

    text_parts: list[str] = []
    try:
        with pdf:
            page_count = len(pdf)
            for page_number, page in enumerate(pdf, start=1):
                try:
                    text_parts.append(page.get_text("text"))
                except Exception as exc:
                    raise DocumentLoadError(
                        f"Could not extract text from PDF page {page_number} with PyMuPDF: "
                        f"{path}. Original error: {exc}"
                    ) from exc
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(
            f"Could not read PDF with PyMuPDF: {path}. Original error: {exc}"
        ) from exc

    return "\n".join(text_parts), page_count


def _extract_pdf_text_with_ocr(path: Path, ocr_service: OCRService) -> str:
    try:
        import fitz
        import numpy as np
    except ImportError as exc:
        raise DocumentLoadError(
            "PyMuPDF and NumPy are required for PDF OCR fallback. Install packages: pymupdf numpy"
        ) from exc

    text_parts: list[str] = []
    with fitz.open(path) as pdf:
        for page in pdf:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height,
                pixmap.width,
                pixmap.n,
            )
            text_parts.append(ocr_service.extract_text(image))
    return "\n\n".join(part for part in text_parts if part.strip())


def _file_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "modified_time": stat.st_mtime,
    }
