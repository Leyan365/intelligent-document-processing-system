"""Hybrid regex and spaCy information extraction."""

import re
from functools import lru_cache
from typing import Any


FIELD_NAMES = ("invoice_number", "date", "amount", "supplier")

INVOICE_NUMBER_PATTERN = re.compile(
    r"\b(?:invoice\s*(?:no|number|#)?\s*[:\-]?\s*)?((?:INV|PO)-?\d{3,})\b",
    re.IGNORECASE,
)
AMOUNT_PATTERN = re.compile(
    r"(?:total|amount|balance\s+due|grand\s+total)?\s*[:\-]?\s*(\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\$?\s*\d+(?:\.\d{2})?)",
    re.IGNORECASE,
)
DATE_PATTERN = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
SUPPLIER_LABEL_PATTERN = re.compile(
    r"\b(?:supplier|vendor|from|bill\s+from)\s*[:\-]\s*(.+)",
    re.IGNORECASE,
)


def extract_fields(text: str) -> dict[str, str | None]:
    """Extract common business-document fields from text."""
    if not text:
        return _empty_fields()

    doc = _parse_with_spacy(text)
    return {
        "invoice_number": _first_regex_group(INVOICE_NUMBER_PATTERN, text),
        "date": _extract_date(text, doc),
        "amount": _extract_amount(text),
        "supplier": _extract_supplier(text, doc),
    }


class InformationExtractor:
    """Small wrapper around the field extraction function."""

    def extract(self, text: str, document_type: str | None = None) -> dict[str, str | None]:
        return extract_fields(text)


def _empty_fields() -> dict[str, None]:
    return {field: None for field in FIELD_NAMES}


def _first_regex_group(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    value = match.group(1) if match.lastindex else match.group(0)
    return _clean_value(value)


def _extract_date(text: str, doc: Any | None) -> str | None:
    if doc is not None:
        for entity in doc.ents:
            if entity.label_ == "DATE":
                return _clean_value(entity.text)
    return _first_regex_group(DATE_PATTERN, text)


def _extract_amount(text: str) -> str | None:
    candidates = []
    for match in AMOUNT_PATTERN.finditer(text):
        value = _clean_value(match.group(1))
        if value and any(char.isdigit() for char in value):
            candidates.append(value)
    return candidates[-1] if candidates else None


def _extract_supplier(text: str, doc: Any | None) -> str | None:
    labeled_supplier = _extract_supplier_from_label(text)
    if labeled_supplier:
        return labeled_supplier

    if doc is not None:
        for entity in doc.ents:
            if entity.label_ == "ORG":
                return _clean_value(entity.text)
    return None


def _extract_supplier_from_label(text: str) -> str | None:
    for line in text.splitlines():
        match = SUPPLIER_LABEL_PATTERN.search(line)
        if match:
            value = re.split(r"\s{2,}|\t|,?\s+(?:date|invoice|total)\b", match.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
            return _clean_value(value)
    return None


def _clean_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip(" .,:;-")
    return cleaned or None


def _parse_with_spacy(text: str) -> Any | None:
    nlp = _load_spacy_model()
    return nlp(text) if nlp is not None else None


@lru_cache(maxsize=1)
def _load_spacy_model() -> Any | None:
    try:
        import spacy
    except ImportError:
        return None

    for model_name in ("en_core_web_sm", "en_core_web_md"):
        try:
            return spacy.load(model_name)
        except OSError:
            continue
    return None


if __name__ == "__main__":
    sample_text = """
    Supplier: Acme Office Supplies
    Invoice No: INV-123
    Date: 2026-05-05
    Total Amount: $1,250.00
    """
    print(extract_fields(sample_text))
