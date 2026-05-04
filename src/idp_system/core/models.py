"""Core data models for local document processing."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4


class DocumentType(str, Enum):
    """Document formats accepted by the IDP pipeline."""

    PDF = "pdf"
    IMAGE = "image"
    TXT = "txt"
    DOCX = "docx"
    MD = "md"
    HTML = "html"
    JSON = "json"
    CSV = "csv"


class ProcessingStatus(str, Enum):
    """Lifecycle states for document processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class Document:
    """Represents an uploaded or loaded source document."""

    title: str
    content: str
    source: str
    doc_type: DocumentType
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, object] = field(default_factory=dict)
    status: ProcessingStatus = ProcessingStatus.PENDING
    extraction_method: str | None = None

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        return len(self.content)

    @classmethod
    def empty_from_path(cls, path: str | Path, doc_type: DocumentType) -> "Document":
        path = Path(path)
        return cls(
            title=path.stem,
            content="",
            source=str(path),
            doc_type=doc_type,
            metadata={"filename": path.name, "extension": path.suffix},
        )


@dataclass(slots=True)
class TextChunk:
    """A chunk of document text for later extraction and search phases."""

    document_id: str
    content: str
    start_index: int
    end_index: int
    chunk_index: int
    id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.content) // 4) if self.content else 0
