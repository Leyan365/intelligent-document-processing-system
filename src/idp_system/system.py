"""Top-level orchestrator for the local IDP system."""

from pathlib import Path
from typing import Iterable

from .core.models import Document, DocumentType
from .pipeline.classifier import DocumentClassifier
from .pipeline.extractor import InformationExtractor
from .pipeline.loader import DocumentLoaderRouter
from .pipeline.search import SemanticSearchService


class IDPSystem:
    """Coordinates the in-memory document processing pipeline."""

    def __init__(
        self,
        loader: DocumentLoaderRouter | None = None,
        classifier: DocumentClassifier | None = None,
        extractor: InformationExtractor | None = None,
        search_service: SemanticSearchService | None = None,
    ) -> None:
        self.loader = loader or DocumentLoaderRouter()
        self.classifier = classifier or DocumentClassifier()
        self.extractor = extractor or InformationExtractor()
        self.search_service = search_service or SemanticSearchService()
        self.documents: dict[str, Document] = {}
        self.processed_documents: dict[str, dict[str, object]] = {}

    def load_document(self, source: str | Path) -> Document:
        """Load one document through the router."""
        document = self.loader.load(source)
        self.documents[document.id] = document
        return document

    def load_documents(self, sources: Iterable[str | Path]) -> list[Document]:
        """Load multiple documents through the router."""
        return [self.load_document(source) for source in sources]

    def process_document(self, source: str | Path) -> dict[str, object]:
        """Load, classify, extract fields, store, and index one document."""
        document = self.load_document(source)
        classification = _classify_document(self.classifier, document.content)
        document_type = str(classification["label"])
        fields = self.extractor.extract(document.content, document_type)

        processed_document = {
            "id": document.id,
            "text": document.content,
            "type": document_type,
            "confidence": classification.get("confidence"),
            "confidence_source": classification.get("confidence_source"),
            "fields": fields,
        }

        self.processed_documents[document.id] = processed_document
        search_text = _build_search_text(document_type, fields, document.content)

        self.search_service.add_documents(
            [
                {
                    "id": document.id,
                    "text": search_text,
                    "type": document_type,
                    "confidence": classification.get("confidence"),
                    "confidence_source": classification.get("confidence_source"),
                    "fields": fields,
                    "source": document.source,
                }
            ]
        )
        return processed_document

    def search(self, query: str, k: int = 5) -> list[dict[str, object]]:
        """Search processed documents by semantic similarity."""
        return self.search_service.search(query, k=k)


def _build_search_text(
    document_type: str,
    fields: dict[str, object],
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


def _field_value(value: object) -> str:
    return "" if value in (None, "") else str(value)


def _classify_document(
    classifier: DocumentClassifier,
    text: str,
) -> dict[str, object]:
    if hasattr(classifier, "classify_with_confidence"):
        return classifier.classify_with_confidence(text)

    return {
        "label": classifier.classify(text),
        "confidence": None,
        "confidence_source": None,
    }


if __name__ == "__main__":
    class ExampleLoader:
        def load(self, source: str | Path) -> Document:
            return Document(
                title="sample_invoice",
                content="Supplier: Acme Office Supplies\nInvoice No: INV-123\nDate: 2026-05-05\nTotal Amount: $1,250.00",
                source=str(source),
                doc_type=DocumentType.PDF,
            )

    class ExampleSearchService:
        def __init__(self) -> None:
            self.documents: list[dict[str, object]] = []

        def add_documents(self, documents: list[dict[str, object]]) -> None:
            self.documents.extend(documents)

        def search(self, query: str, k: int = 5) -> list[dict[str, object]]:
            return self.documents[:k]

    system = IDPSystem(loader=ExampleLoader(), search_service=ExampleSearchService())
    print(system.process_document("sample.pdf"))
