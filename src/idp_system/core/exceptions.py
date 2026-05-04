"""Custom exceptions for the IDP system."""


class IDPSystemError(Exception):
    """Base exception for IDP system errors."""


class DocumentLoadError(IDPSystemError):
    """Raised when a document cannot be loaded."""


class UnsupportedDocumentTypeError(DocumentLoadError):
    """Raised when no loader supports the source document type."""
