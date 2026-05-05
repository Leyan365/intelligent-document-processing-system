"""Hybrid regex and spaCy information extraction."""

import re
from datetime import datetime
from functools import lru_cache
from typing import Any


FIELD_NAMES = ("invoice_number", "date", "amount", "supplier")

INVOICE_NUMBER_PATTERN = re.compile(
    r"\b(?:(?:invoice|order|po)\s*(?:no|number|#)?\s*[:\-]?\s*)?((?:INV|PO)-?\d{3,})\b",
    re.IGNORECASE,
)
AMOUNT_PATTERN = re.compile(
    r"(?:total|amount|balance\s+due|grand\s+total)?\s*[:\-]?\s*(\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\$?\s*\d+(?:\.\d{2})?)",
    re.IGNORECASE,
)
LABELED_AMOUNT_PATTERN = re.compile(
    r"\b(?:grand\s+total|net\s+total|amount\s+due|balance\s+due|total\s+amount|total|amount)\b"
    r"\s*[:\-]?\s*(\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\$?\s*\d+(?:\.\d{2})?)",
    re.IGNORECASE,
)
DATE_PATTERN = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
    r"\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*-?\d{4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|"
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b",
    re.IGNORECASE,
)
LABELED_DATE_PATTERN = re.compile(
    r"\b(?:invoice\s+date|order\s+date|issued\s+date|po\s+date|date)\b\s*[:\-]?\s*"
    r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
    r"\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*-?\d{4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|"
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})",
    re.IGNORECASE,
)
SUPPLIER_LABEL_PATTERN = re.compile(
    r"\b(?:supplier\s+name|supplier|vendor|from|bill\s+from)\s*[:\-]\s*(.+)",
    re.IGNORECASE,
)
SUPPLIER_CODE_PATTERN = re.compile(
    r"\bsupplier\s+\d{2,}\s+(.+?)(?=\s+(?:supplier\s+address|address|ship\s+to|buyer|delivery\s+address|order\s+number|order\s+date)\b|$)",
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
    labeled_date = _first_valid_date(LABELED_DATE_PATTERN, text)
    if labeled_date:
        return labeled_date

    if doc is not None:
        for entity in doc.ents:
            if entity.label_ == "DATE" and _is_plausible_date(entity.text):
                return _clean_value(entity.text)
    return _first_valid_date(DATE_PATTERN, text)


def _extract_amount(text: str) -> str | None:
    labeled_candidates = _amount_candidates(LABELED_AMOUNT_PATTERN, text, require_label=False)
    if labeled_candidates:
        return max(labeled_candidates, key=lambda item: item[0])[1]

    candidates = _amount_candidates(AMOUNT_PATTERN, text, require_label=True)
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _amount_candidates(
    pattern: re.Pattern[str],
    text: str,
    require_label: bool,
) -> list[tuple[float, str]]:
    candidates: list[tuple[float, str]] = []
    for match in pattern.finditer(text):
        value = _clean_value(match.group(1))
        amount = _parse_amount(value)
        context = text[max(0, match.start() - 40) : match.end() + 20].lower()
        has_label = any(
            label in context
            for label in ("total", "amount", "balance due", "grand total", "net total")
        )
        if value and amount is not None and not _looks_like_noise_context(context):
            if has_label or (not require_label and amount >= 10):
                if amount >= 10 or has_label:
                    candidates.append((amount, value))
    return candidates


def _extract_supplier(text: str, doc: Any | None) -> str | None:
    labeled_supplier = _extract_supplier_from_label(text)
    if labeled_supplier:
        return labeled_supplier

    if doc is not None:
        for entity in doc.ents:
            if entity.label_ == "ORG" and _is_plausible_supplier(entity.text):
                return _clean_value(entity.text)
    return None


def _extract_supplier_from_label(text: str) -> str | None:
    compact_text = " ".join(text.split())
    code_match = SUPPLIER_CODE_PATTERN.search(compact_text)
    if code_match:
        value = _clean_value(code_match.group(1))
        if _is_plausible_supplier(value):
            return value

    for line in text.splitlines():
        match = SUPPLIER_LABEL_PATTERN.search(line)
        if match:
            value = re.split(
                r"\s{2,}|\t|,?\s+(?:date|invoice|total|supplier\s+address|address|ship\s+to|buyer|delivery\s+address|order\s+number|order\s+date)\b",
                match.group(1),
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            value = _clean_value(value)
            if _is_plausible_supplier(value):
                return value
    return None


def _clean_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip(" .,:;-")
    return cleaned or None


def _first_valid_date(pattern: re.Pattern[str], text: str) -> str | None:
    for match in pattern.finditer(text):
        value = match.group(1) if match.lastindex else match.group(0)
        if _is_plausible_date(value):
            return _clean_value(value)
    return None


def _is_plausible_date(value: str | None) -> bool:
    cleaned = _clean_value(value)
    if not cleaned or len(cleaned) > 30:
        return False
    if re.search(r"[A-Z]{2,}/[A-Z]{2,}/", cleaned, re.IGNORECASE):
        return False

    formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%d-%b-%Y",
        "%d-%B-%Y",
        "%d/%m/%y",
        "%m/%d/%y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
        "%d %B %Y",
        "%d %b %Y",
    )
    for date_format in formats:
        try:
            parsed = datetime.strptime(cleaned, date_format)
            return 1990 <= parsed.year <= 2100
        except ValueError:
            continue
    return False


def _parse_amount(value: str | None) -> float | None:
    cleaned = _clean_value(value)
    if not cleaned:
        return None
    normalized = cleaned.replace("$", "").replace(",", "").replace(" ", "")
    try:
        return float(normalized)
    except ValueError:
        return None


def _looks_like_noise_context(context: str) -> bool:
    return any(
        marker in context
        for marker in ("doc. ref", "ref. no", "revision", "rev. no", "page", "attachment")
    )


def _is_plausible_supplier(value: str | None) -> bool:
    cleaned = _clean_value(value)
    if not cleaned or len(cleaned) < 4:
        return False
    if re.fullmatch(r"[A-Z0-9/().\-]+", cleaned, re.IGNORECASE):
        return False
    lowered = cleaned.lower()
    if any(
        marker in lowered
        for marker in (
            "export processing zone",
            "processing zone",
            "industrial zone",
            "ship to",
            "buyer",
            "delivery address",
            "address",
        )
    ):
        return False
    alnum_chars = [char for char in cleaned if char.isalnum()]
    if not alnum_chars:
        return False
    digit_ratio = sum(char.isdigit() for char in alnum_chars) / len(alnum_chars)
    letter_count = sum(char.isalpha() for char in alnum_chars)
    return digit_ratio < 0.35 and letter_count >= 3


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
    po_text = (
        "Pro-forma Purchase Order Supplier 116451 Screenline (Pvt) Ltd "
        "Supplier Address No.18/4, Thalwatha, Gonawala Kelaniya "
        "Order Number PO10042153 Order Date 21-Jan-2026 Total 5,746.60 "
        "Ship To Ansell Lanka (Pvt) Ltd Biyagama Export Processing Zone"
    )
    print(extract_fields(po_text))
    sample_text = """
    Supplier: Acme Office Supplies
    Invoice No: INV-123
    Date: 2026-05-05
    Total Amount: $1,250.00
    """
    print(extract_fields(sample_text))
    noisy_text = """
    Doc. Ref. No.: LRP/GP/ST/002/02(C)
    Rev. No.: 08
    """
    print(extract_fields(noisy_text))
