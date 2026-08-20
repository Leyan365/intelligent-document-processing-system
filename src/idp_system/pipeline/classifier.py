"""Local document classification with TF-IDF and Logistic Regression."""

from pathlib import Path
import re

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# simple in-process cache for the default model
_cached_model: Pipeline | None = None
DEFAULT_MODEL_PATH = Path("models") / "document_classifier.joblib"

SUPPORTED_DOCUMENT_TYPES = ("invoice", "receipt", "purchase_order")


EXAMPLE_TEXTS = [
    "Invoice number INV-1001 issued to customer. Subtotal, tax, and total amount due.",
    "Tax invoice from supplier with billing address, invoice date, and payment terms.",
    "Receipt for cash payment. Items purchased, amount paid, balance, and cashier name.",
    "Sales receipt showing store name, transaction id, payment method, and change.",
    "Purchase order PO-2205 requesting office supplies with quantity and unit price.",
    "Purchase order approval for vendor delivery with requested items and shipping date.",
]

EXAMPLE_LABELS = [
    "invoice",
    "invoice",
    "receipt",
    "receipt",
    "purchase_order",
    "purchase_order",
]


def train_classifier(texts: list[str], labels: list[str]) -> Pipeline:
    """Train a local TF-IDF and Logistic Regression document classifier."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if not texts:
        raise ValueError("at least one training example is required")

    unknown_labels = sorted(set(labels) - set(SUPPORTED_DOCUMENT_TYPES))
    if unknown_labels:
        raise ValueError(f"unsupported labels: {', '.join(unknown_labels)}")

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )
    model.fit(texts, labels)
    return model


def predict_document_type(text: str, model: Pipeline | None = None) -> str:
    """Predict one of invoice, receipt, or purchase_order for input text."""
    if not text or not text.strip():
        raise ValueError("text is required for prediction")

    heuristic_type = heuristic_document_type(text)
    if heuristic_type:
        return heuristic_type

    global _cached_model

    if model is None:
        if _cached_model is None:
            _cached_model = load_default_model()
        model = _cached_model

    return str(model.predict([text])[0])


def predict_document_type_with_confidence(
    text: str,
    model: Pipeline | None = None,
) -> dict[str, float | str | None]:
    """Predict document type and expose model confidence when available."""
    if not text or not text.strip():
        raise ValueError("text is required for prediction")

    heuristic_type = heuristic_document_type(text)
    if heuristic_type:
        return {
            "label": heuristic_type,
            "confidence": None,
            "confidence_source": "heuristic",
        }

    global _cached_model

    if model is None:
        if _cached_model is None:
            _cached_model = load_default_model()
        model = _cached_model

    label = str(model.predict([text])[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(label)])

    return {
        "label": label,
        "confidence": confidence,
        "confidence_source": "model" if confidence is not None else None,
    }


def heuristic_document_type(text: str) -> str | None:
    """Return a type for strong business-document signals, else None."""
    normalized = " ".join(text.lower().split())

    purchase_order_score = sum(
        signal in normalized
        for signal in (
            "pro-forma purchase order",
            "purchase order",
            "order number",
        )
    )
    if re_search(r"\bpo-?\d{3,}\b", normalized):
        purchase_order_score += 1
    if purchase_order_score >= 1:
        return "purchase_order"

    receipt_score = sum(
        signal in normalized
        for signal in (
            "receipt",
            "receipt #",
            "receipt date",
            "cashier",
            "paid amount",
            "amount paid",
            "cash bill",
            "change",
            "total items",
            "total qty",
            "total oty",
            "receipt no",
            "payment mode",
            "tender",
        )
    )

    invoice_score = sum(
        signal in normalized
        for signal in (
            "invoice #",
            "invoice no",
            "invoice number",
            "invoice date",
            "balance due",
            "amount due",
            "subtotal",
        )
    )
    if "bill to" in normalized and "ship to" in normalized:
        invoice_score += 2
    if "subtotal" in normalized and "total" in normalized:
        invoice_score += 1
    if receipt_score >= 1:
        return "receipt"

    if invoice_score >= 1 and not _looks_like_receipt_tax_invoice(normalized, receipt_score):
        return "invoice"

    if "tax invoice" in normalized:
        invoice_score += 1
    if invoice_score >= 1:
        return "invoice"

    return None


def _looks_like_receipt_tax_invoice(normalized_text: str, receipt_score: int) -> bool:
    """Keep POS-style tax invoices classified as receipts."""
    return "tax invoice" in normalized_text and receipt_score >= 1


def re_search(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def save_model(model: Pipeline, path: str | Path) -> Path:
    """Save a trained classifier locally with joblib."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def load_model(path: str | Path) -> Pipeline:
    """Load a trained classifier from local disk."""
    return joblib.load(Path(path))


def load_default_model() -> Pipeline:
    """Load the trained local classifier, falling back to the example model."""
    model_path = DEFAULT_MODEL_PATH
    if model_path.exists():
        try:
            return load_model(model_path)
        except Exception as exc:
            print(
                f"Warning: failed to load {model_path}: {type(exc).__name__}: {exc}. "
                "Using example classifier fallback."
            )
    return train_example_classifier()


def train_example_classifier() -> Pipeline:
    """Train a tiny in-file classifier for smoke testing."""
    return train_classifier(EXAMPLE_TEXTS, EXAMPLE_LABELS)


class DocumentClassifier:
    """Thin object-oriented wrapper around the local classifier pipeline."""

    def __init__(self, model: Pipeline | None = None) -> None:
        global _cached_model
        if model is not None:
            self.model = model
        else:
            if _cached_model is None:
                _cached_model = load_default_model()
            self.model = _cached_model

    def classify(self, text: str) -> str:
        return predict_document_type(text, self.model)

    def classify_with_confidence(self, text: str) -> dict[str, float | str | None]:
        return predict_document_type_with_confidence(text, self.model)

    def save(self, path: str | Path) -> Path:
        return save_model(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> "DocumentClassifier":
        return cls(load_model(path))


if __name__ == "__main__":
    classifier = train_example_classifier()
    sample = "Invoice INV-204 contains supplier name, due date, tax, and total amount."
    print(predict_document_type(sample, classifier))
    receipt_sample = (
        "Company Name: Quantum Logic Solutions RECEIPT Address Bill To "
        "Receipt # 100 Receipt Date 07/05/2026 TOTAL Rs. 13,500.00"
    )
    print(predict_document_type(receipt_sample, classifier))
    po_sample = (
        "Pro-forma Purchase Order Supplier 116451 Screenline (Pvt) Ltd "
        "Supplier Address No.18/4, Thalwatha, Gonawala Kelaniya "
        "Order Number PO10042153 Order Date 21-Jan-2026 Total 5,746.60"
    )
    print(predict_document_type(po_sample, classifier))
