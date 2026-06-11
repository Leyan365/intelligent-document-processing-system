"""Hybrid regex and spaCy information extraction."""

import re
from datetime import datetime
from functools import lru_cache
from typing import Any


FIELD_NAMES = ("invoice_number", "date", "amount", "supplier")

DATE_VALUE_PATTERN = (
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|"
    r"\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*-?\d{4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|"
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}"
)
AMOUNT_VALUE_PATTERN = (
    r"(?:Rs\.?|RM|\$)?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})|"
    r"(?:Rs\.?|RM|\$)?\s*\d+(?:\.\d{2})"
)

INVOICE_NUMBER_PATTERN = re.compile(
    r"\b(?:invoice\s*(?:no|number|#)?|inv)\s*[:#\-]?\s*((?:INV[-\s]?)?\d{3,})\b",
    re.IGNORECASE,
)
GENERIC_NUMBER_PATTERN = re.compile(
    r"\b(?:(?:invoice|order|po)\s*(?:no|number|#)?\s*[:\-]?\s*)?((?:INV|PO)-?\d{3,})\b",
    re.IGNORECASE,
)
PO_NUMBER_PATTERNS = (
    re.compile(r"\bpo\s*number\s*[:#\-]?\s*([A-Z]*\d{4,}(?:-\d+)?)\b", re.IGNORECASE),
    re.compile(r"\border\s+number\s*[:#\-]?\s*((?:PO\s*)?[-#]?\s*\d{3,})\b", re.IGNORECASE),
    re.compile(
        r"\bpurchase\s+order\s*(?:no|number|#)?\s*[:#\-]?\s*((?:PO\s*)?[-#]?\s*\d{3,})\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bpo\s*[#:\-]\s*(\d{3,})\b", re.IGNORECASE),
    re.compile(r"\b(PO[-#]?\d{3,}(?:-\d+)?)\b", re.IGNORECASE),
)
LABELED_DATE_PATTERN = re.compile(
    rf"\b(?:invoice\s+date|order\s+date|issued\s+date|po\s+date|date)\b\s*[:\-]?\s*({DATE_VALUE_PATTERN})",
    re.IGNORECASE,
)
DATE_PATTERN = re.compile(rf"\b({DATE_VALUE_PATTERN})\b", re.IGNORECASE)
PO_DATE_PATTERNS = (
    re.compile(rf"\bpo\s+creation\s+date\s*[:\-]?\s*({DATE_VALUE_PATTERN})", re.IGNORECASE),
    re.compile(rf"\border\s+date\s*[:\-]?\s*({DATE_VALUE_PATTERN})", re.IGNORECASE),
    re.compile(rf"\bdate\s*[:\-]?\s*({DATE_VALUE_PATTERN})", re.IGNORECASE),
)
LABELED_AMOUNT_PATTERN = re.compile(
    rf"\b(?:grand\s+total|net\s+total|amount\s+due|balance\s+due|total\s+amount|total|amount)\b"
    rf"\s*[:\-]?\s*({AMOUNT_VALUE_PATTERN})",
    re.IGNORECASE,
)
INVOICE_AMOUNT_LABELS = (
    r"balance\s+due",
    r"grand\s+total",
    r"amount\s+due",
    r"total\s+amount",
    r"total",
)
PO_AMOUNT_PATTERN = re.compile(
    rf"\b(?:grand\s+total|order\s+total|net\s+total|total\s+value|total\s+amount|total)\b"
    rf"\s*[:\-]?\s*({AMOUNT_VALUE_PATTERN})",
    re.IGNORECASE,
)
RECEIPT_AMOUNT_PATTERN = re.compile(
    rf"\b(?:grand\s+total|net\s+total|total\s+amt\s+payable|totalamt\s+payable|"
    rf"total\s+amount|amount\s+payable|paid\s+amount|balance\s+due|amount|total)\b"
    rf"\s*[:\-]?\s*({AMOUNT_VALUE_PATTERN})",
    re.IGNORECASE,
)
RECEIPT_AMOUNT_LABELS = (
    r"grand\s+total",
    r"net\s+total",
    r"total\s+amt\s+payable",
    r"totalamt\s+payable",
    r"total\s+amt(?:\s+incl\.?\s+gst)?",
    r"total\s+amount",
    r"amount\s+payable",
    r"paid\s+amount",
    r"total",
)
SUPPLIER_LABEL_PATTERN = re.compile(
    r"\b(?:supplier\s+name|supplier|vendor|from|bill\s+from|company\s+name)\s*[:\-]\s*(.+)",
    re.IGNORECASE,
)
SUPPLIER_CODE_PATTERN = re.compile(
    r"\bsupplier\s+\d{2,}\s+(.+?)(?=\s+(?:supplier\s+address|address|ship\s+to|buyer|delivery\s+address|order\s+number|order\s+date)\b|$)",
    re.IGNORECASE,
)

BAD_SUPPLIER_MARKERS = (
    "total",
    "net total",
    "amount",
    "terms",
    "conditions",
    "payment",
    "cashier",
    "change",
    "gst",
    "gst no",
    "trn",
    "tax invoice",
    "invoice",
    "invoice #",
    "company name",
    "counter",
    "ship to",
    "bill to",
    "customer name",
    "buyer",
    "qty",
    "uom",
    "customer payment",
    "customer's payment",
    "contact",
    "delivery address",
    "address",
    "last modified",
    "phones, technology",
)
SUPPLIER_STOP_MARKERS = (
    "supplier address",
    "address",
    "receipt",
    "bill to",
    "customer name",
    "date",
    "gst",
    "gst no",
    "trn",
    "tax invoice",
    "counter",
    "qty",
    "uom",
    "customer payment",
    "customer's payment",
    "contact",
    "terms",
    "cashier",
    "fax",
    "vat",
    "svat",
    "delivery address",
    "ship to",
    "buyer",
    "order number",
    "order date",
    "po creation date",
)


def extract_fields(text: str, document_type: str | None = None) -> dict[str, str | None]:
    """Extract common business-document fields from text."""
    if not text:
        return _empty_fields()

    doc_type = _normalize_document_type(document_type)
    doc = _parse_with_spacy(text)

    if doc_type == "purchase_order":
        return _extract_purchase_order_fields(text)
    if doc_type == "receipt":
        return _extract_receipt_fields(text)
    if doc_type == "invoice":
        return _extract_invoice_fields(text, doc)

    return {
        "invoice_number": _first_regex_group(GENERIC_NUMBER_PATTERN, text),
        "date": _extract_date(text, doc),
        "amount": _extract_amount(text),
        "supplier": _extract_supplier(text, doc),
    }


class InformationExtractor:
    """Small wrapper around the field extraction function."""

    def extract(self, text: str, document_type: str | None = None) -> dict[str, str | None]:
        return extract_fields(text, document_type)


def _extract_purchase_order_fields(text: str) -> dict[str, str | None]:
    return {
        "invoice_number": _extract_po_number(text),
        "date": _extract_po_date(text),
        "amount": _extract_po_amount(text),
        "supplier": _extract_po_supplier(text),
    }


def _extract_receipt_fields(text: str) -> dict[str, str | None]:
    return {
        "invoice_number": _first_regex_group(INVOICE_NUMBER_PATTERN, text),
        "date": _extract_receipt_date(text),
        "amount": _extract_receipt_amount(text),
        "supplier": _extract_receipt_supplier(text),
    }


def _extract_invoice_fields(text: str, doc: Any | None) -> dict[str, str | None]:
    return {
        "invoice_number": _first_regex_group(INVOICE_NUMBER_PATTERN, text)
        or _first_regex_group(GENERIC_NUMBER_PATTERN, text),
        "date": _extract_date(text, doc),
        "amount": _extract_amount(text),
        "supplier": _extract_invoice_supplier(text, doc),
    }


def _empty_fields() -> dict[str, None]:
    return {field: None for field in FIELD_NAMES}


def _normalize_document_type(document_type: str | None) -> str | None:
    if document_type is None:
        return None
    normalized = str(document_type).strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in {"po", "purchaseorder"}:
        return "purchase_order"
    return normalized


def _first_regex_group(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    value = match.group(1) if match.lastindex else match.group(0)
    return _clean_value(value)


def _extract_po_number(text: str) -> str | None:
    for pattern in PO_NUMBER_PATTERNS:
        value = _first_regex_group(pattern, text)
        if value:
            return re.sub(r"\s+", "", value).lstrip("#-")
    return None


def _extract_date(text: str, doc: Any | None) -> str | None:
    labeled_date = _first_valid_date(LABELED_DATE_PATTERN, text)
    if labeled_date:
        return labeled_date

    if doc is not None:
        for entity in doc.ents:
            if entity.label_ == "DATE" and _is_plausible_date(entity.text):
                return _clean_value(entity.text)
    return _first_valid_date(DATE_PATTERN, text)


def _extract_po_date(text: str) -> str | None:
    for pattern in PO_DATE_PATTERNS:
        value = _first_valid_date(pattern, text)
        if value:
            return value
    return _first_valid_date(DATE_PATTERN, _without_last_modified_lines(text))


def _extract_receipt_date(text: str) -> str | None:
    return _first_valid_date(DATE_PATTERN, text)


def _extract_amount(text: str) -> str | None:
    labeled_candidates = _amount_candidates(LABELED_AMOUNT_PATTERN, text)
    if labeled_candidates:
        return max(labeled_candidates, key=lambda item: item[0])[1]
    return _nearest_amount_for_labels(text, INVOICE_AMOUNT_LABELS)


def _extract_po_amount(text: str) -> str | None:
    candidates = _amount_candidates(PO_AMOUNT_PATTERN, text)
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _extract_receipt_amount(text: str) -> str | None:
    priority_amount = _nearest_amount_for_labels(text, RECEIPT_AMOUNT_LABELS)
    if priority_amount:
        return priority_amount
    candidates = _amount_candidates(RECEIPT_AMOUNT_PATTERN, text)
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _amount_candidates(pattern: re.Pattern[str], text: str) -> list[tuple[float, str]]:
    candidates: list[tuple[float, str]] = []
    for match in pattern.finditer(text):
        value = _clean_value(match.group(1))
        amount = _parse_amount(value)
        context = text[max(0, match.start() - 50) : match.end() + 30].lower()
        if (
            value
            and amount is not None
            and not _looks_like_noise_context(context)
            and not _looks_like_bad_amount_context(context)
        ):
            candidates.append((amount, value))
    return candidates


def _nearest_amount_for_labels(text: str, label_patterns: tuple[str, ...]) -> str | None:
    for label_pattern in label_patterns:
        pattern = re.compile(rf"\b{label_pattern}\b", re.IGNORECASE)
        label_candidates: list[tuple[int, float, str]] = []
        for label_match in pattern.finditer(text):
            start = max(0, label_match.start() - 90)
            end = min(len(text), label_match.end() + 120)
            window = text[start:end]
            for amount_match in re.finditer(AMOUNT_VALUE_PATTERN, window, flags=re.IGNORECASE):
                value = _clean_value(amount_match.group(0))
                amount = _parse_amount(value)
                absolute_start = start + amount_match.start()
                context = text[max(0, absolute_start - 30) : start + amount_match.end() + 30].lower()
                if (
                    value
                    and amount is not None
                    and not _looks_like_noise_context(context)
                    and not _looks_like_bad_amount_context(context)
                ):
                    distance = min(
                        abs(absolute_start - label_match.start()),
                        abs(absolute_start - label_match.end()),
                    )
                    label_candidates.append((distance, amount, value))
        if label_candidates:
            return max(label_candidates, key=lambda item: (-item[0], item[1]))[2]
    return None


def _extract_supplier(text: str, doc: Any | None) -> str | None:
    labeled_supplier = _extract_supplier_from_label(text)
    if labeled_supplier:
        return labeled_supplier

    if doc is not None:
        for entity in doc.ents:
            if entity.label_ == "ORG" and _is_plausible_supplier(entity.text):
                return _clean_supplier(entity.text)
    return _extract_top_merchant_line(text)


def _extract_invoice_supplier(text: str, doc: Any | None) -> str | None:
    supplier = _extract_supplier_from_label(text)
    if supplier:
        return supplier
    inline_supplier = _extract_invoice_supplier_after_number(text)
    if inline_supplier:
        return inline_supplier
    top_line = _extract_top_merchant_line(text)
    if top_line:
        return top_line
    if doc is not None:
        for entity in doc.ents:
            if entity.label_ == "ORG" and _is_plausible_supplier(entity.text):
                return _clean_supplier(entity.text)
    return None


def _extract_receipt_supplier(text: str) -> str | None:
    supplier = _extract_supplier_from_label(text)
    if supplier:
        return supplier
    return _extract_top_merchant_line(text)


def _extract_invoice_supplier_after_number(text: str) -> str | None:
    compact_text = " ".join(text.split())
    match = INVOICE_NUMBER_PATTERN.search(compact_text)
    if not match:
        return None

    remainder = compact_text[match.end() : match.end() + 140]
    value = re.split(
        r"\b(?:bill\s+to|ship\s+to|customer\s+name|buyer|contact|invoice\s+date|date|balance\s+due|grand\s+total|total)\b",
        remainder,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    supplier = _clean_supplier(value)
    return supplier if _is_plausible_supplier(supplier) else None


def _extract_po_supplier(text: str) -> str | None:
    compact_text = " ".join(text.split())
    code_match = SUPPLIER_CODE_PATTERN.search(compact_text)
    if code_match:
        value = _clean_supplier(code_match.group(1))
        if _is_plausible_supplier(value):
            return value

    supplier_match = re.search(
        r"\bsupplier\s*:\s*(.+?)(?=\s+(?:supplier\s+address|address|contact|fax|vat|delivery\s+address|ship\s+to|buyer|po\s+number|order\s+number)\b|$)",
        compact_text,
        flags=re.IGNORECASE,
    )
    if supplier_match:
        value = _clean_supplier(supplier_match.group(1))
        if _is_plausible_supplier(value):
            return value

    return _extract_supplier_from_label(text)


def _extract_supplier_from_label(text: str) -> str | None:
    compact_text = " ".join(text.split())
    code_match = SUPPLIER_CODE_PATTERN.search(compact_text)
    if code_match:
        value = _clean_supplier(code_match.group(1))
        if _is_plausible_supplier(value):
            return value

    for line in text.splitlines():
        match = SUPPLIER_LABEL_PATTERN.search(line)
        if match:
            value = _clean_supplier(match.group(1))
            if _is_plausible_supplier(value):
                return value
    return None


def _extract_top_merchant_line(text: str) -> str | None:
    lines = [_clean_value(line) for line in text.splitlines()]
    meaningful = [line for line in lines if line]
    for line in meaningful[:12]:
        candidate = _clean_supplier(line)
        if _is_plausible_supplier(candidate):
            return candidate
    return None


def _clean_supplier(value: str | None) -> str | None:
    cleaned = _clean_value(value)
    if not cleaned:
        return None

    earliest_stop: int | None = None
    for marker in SUPPLIER_STOP_MARKERS:
        match = re.search(rf"\b{re.escape(marker)}\b", cleaned, flags=re.IGNORECASE)
        if match:
            earliest_stop = match.start() if earliest_stop is None else min(earliest_stop, match.start())
    if earliest_stop is not None:
        cleaned = cleaned[:earliest_stop]

    cleaned = re.sub(r"\s*-\s*\d{4,}\b.*$", "", cleaned)
    cleaned = re.sub(r"\b\d{4,}\b\s*$", "", cleaned)
    cleaned = _clean_value(cleaned)
    return None if _looks_like_bad_supplier(cleaned) else cleaned


def _looks_like_bad_supplier(value: str | None) -> bool:
    cleaned = _clean_value(value)
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if any(marker in lowered for marker in BAD_SUPPLIER_MARKERS):
        return True
    if len(cleaned) > 80:
        return True
    if re.fullmatch(r"[\d\s.,:/\-]+", cleaned):
        return True
    return False


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
        "%d.%m.%Y",
        "%m.%d.%Y",
        "%d-%b-%Y",
        "%d-%B-%Y",
        "%d/%m/%y",
        "%m/%d/%y",
        "%d.%m.%y",
        "%m.%d.%y",
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
    normalized = (
        cleaned.replace("$", "")
        .replace("Rs.", "")
        .replace("Rs", "")
        .replace("RM", "")
        .replace(",", "")
        .replace(" ", "")
    )
    try:
        return float(normalized)
    except ValueError:
        return None


def _looks_like_noise_context(context: str) -> bool:
    return any(
        marker in context
        for marker in (
            "doc. ref",
            "ref. no",
            "revision",
            "rev. no",
            "page",
            "attachment",
            "unit price",
            "last modified",
        )
    )


def _looks_like_bad_amount_context(context: str) -> bool:
    return any(
        re.search(pattern, context, flags=re.IGNORECASE)
        for pattern in (
            r"\btotal\s*(?:qty|quantity|items?)\b",
            r"\bqty\b",
            r"\buom\b",
            r"\bunit\s*amt\b",
            r"\bitem\s*count\b",
            r"\bcounter\s*[:\-]?\s*\d+\b",
            r"\bcashier\s*[:\-]?\s*\d+\b",
            r"\bgst\s*(?:rate)?\s*\d+(?:\.\d+)?\s*%",
            r"\btax\s*(?:rate)?\s*\d+(?:\.\d+)?\s*%",
            r"\btax\s+code\b",
        )
    )


def _without_last_modified_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if "last modified" not in line.lower())


def _is_plausible_supplier(value: str | None) -> bool:
    cleaned = _clean_supplier(value)
    if not cleaned or len(cleaned) < 4:
        return False
    if _looks_like_bad_supplier(cleaned):
        return False
    if " " not in cleaned and re.search(r"[\d/]", cleaned):
        return False
    lowered = cleaned.lower()
    if any(
        marker in lowered
        for marker in (
            "export processing zone",
            "processing zone",
            "industrial zone",
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
    from idp_system.pipeline.classifier import heuristic_document_type

    invoice_text = """
    INVOICE # 6817 SuperStore Bill To: Aaron Hawkins Ship To: Some Address
    Oct 23 2012
    $10,672.30
    Balance Due
    Total
    """
    noisy_receipt_text = """
    GST NO. : 14/02/2018 5:37:44PM TAX INVOICE TRN: CR0005140 COUNTER 4
    CASHIER: 2 QTY UOM UNITAmt Exc. Amt Inc. GST Price Tax Tax Code
    100PLUS LIME 325ML WALK ZRL *Total Qty: 1.00 Total Includes Gst 0%
    Customer's Payment Cash
    """
    receipt_text = """
    Company Name: Quantum Logic Solutions RECEIPT Address: No 12 Main Street
    Date: 07/05/2026
    TOTAL AMT PAYABLE Rs. 13,500.00
    Paid Amount Rs. 13,500.00
    Change Rs. 0.00
    """
    po_text = """
    Last Modified Time 12:01
    PO Number : 5380034370
    PO Creation Date : 25.01.2026
    SUPPLIER: SCREENLINE (PVT) LTD-1007037
    Item Unit Price 17.52
    """
    print(heuristic_document_type(invoice_text))
    print(extract_fields(po_text, "purchase_order"))
    print(extract_fields(noisy_receipt_text, "receipt"))
    print(extract_fields(receipt_text, "receipt"))
    print(extract_fields(invoice_text, "invoice"))
