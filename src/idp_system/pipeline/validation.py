"""Advisory validation checks for the local IDP pipeline."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


STATUS_PASS = "pass"
STATUS_WARNING = "warning"
STATUS_FAIL = "fail"

PIPELINE_PROCESSED = "processed"
PIPELINE_WARNINGS = "processed_with_warnings"
PIPELINE_REVIEW = "needs_review"

FIELD_NAMES = ("invoice_number", "date", "amount", "supplier")
BAD_SUPPLIER_VALUES = {
    "terms",
    "bill to",
    "ship to",
    "address",
    "payment",
    "counter",
    "cashier",
    "gst no",
    "trn",
    "invoice",
}


def validate_text_quality(text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate extracted text quality without blocking downstream processing."""
    metadata = metadata or {}
    cleaned = text or ""
    stripped = cleaned.strip()
    char_count = len(stripped)
    words = re.findall(r"\b[\w'-]+\b", stripped)
    word_count = len(words)
    line_count = len([line for line in stripped.splitlines() if line.strip()])
    non_space_chars = [char for char in stripped if not char.isspace()]
    digit_count = sum(char.isdigit() for char in non_space_chars)
    uppercase_count = sum(char.isupper() for char in non_space_chars)
    symbol_count = sum(
        not char.isalnum() and char not in ".,:/#-&()$%+"
        for char in non_space_chars
    )
    base_count = len(non_space_chars) or 1
    digit_ratio = digit_count / base_count
    uppercase_ratio = uppercase_count / base_count
    symbol_noise_ratio = symbol_count / base_count
    extraction_method = metadata.get("extraction_method")

    warnings: list[str] = []
    critical_warnings: list[str] = []
    status = STATUS_PASS

    if char_count == 0:
        critical_warnings.append("Extracted text is empty.")
    elif char_count < 20 or word_count < 3:
        critical_warnings.append("Extracted text is extremely short.")
    elif char_count < 80 or word_count < 10:
        warnings.append("Extracted text is short; classification and extraction may be less reliable.")

    if symbol_noise_ratio >= 0.45:
        critical_warnings.append("Extracted text has very high symbol noise.")
    elif symbol_noise_ratio >= 0.25:
        warnings.append("Extracted text has elevated symbol noise.")

    if _is_ocr_method(extraction_method) and (char_count < 120 or symbol_noise_ratio >= 0.25):
        warnings.append("OCR-derived text quality is weak; review extracted fields.")

    if critical_warnings:
        status = STATUS_FAIL
    elif warnings:
        status = STATUS_WARNING

    return {
        "status": status,
        "char_count": char_count,
        "word_count": word_count,
        "line_count": line_count,
        "digit_ratio": round(digit_ratio, 4),
        "symbol_noise_ratio": round(symbol_noise_ratio, 4),
        "uppercase_ratio": round(uppercase_ratio, 4),
        "extraction_method": extraction_method,
        "warnings": warnings + critical_warnings,
        "critical_warnings": critical_warnings,
    }


def validate_classification(
    label: str,
    confidence: float | None,
    confidence_source: str | None,
) -> dict[str, Any]:
    """Validate classification confidence and source."""
    warnings: list[str] = []
    critical_warnings: list[str] = []
    status = STATUS_PASS

    if confidence_source == "heuristic":
        return {
            "status": STATUS_PASS,
            "label": label,
            "confidence": confidence,
            "confidence_source": confidence_source,
            "warnings": [],
            "critical_warnings": [],
        }

    if confidence is None:
        warnings.append("Classification confidence is unavailable.")
        status = STATUS_WARNING
    elif confidence >= 0.70:
        status = STATUS_PASS
    elif confidence >= 0.45:
        warnings.append("Classification confidence is moderate; review document type if fields look wrong.")
        status = STATUS_WARNING
    else:
        critical_warnings.append("Classification confidence is very low.")
        status = STATUS_FAIL

    return {
        "status": status,
        "label": label,
        "confidence": confidence,
        "confidence_source": confidence_source,
        "warnings": warnings + critical_warnings,
        "critical_warnings": critical_warnings,
    }


def validate_fields(document_type: str, fields: dict[str, Any]) -> dict[str, Any]:
    """Validate extracted fields with document-type-aware tolerance."""
    normalized_type = _normalize_document_type(document_type)
    field_results: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    critical_warnings: list[str] = []

    for field_name in FIELD_NAMES:
        value = fields.get(field_name)
        result = _validate_field(field_name, value, normalized_type)
        field_results[field_name] = result
        warnings.extend(
            warning
            for warning in result["warnings"]
            if warning not in result["critical_warnings"]
        )
        critical_warnings.extend(result["critical_warnings"])

    status = STATUS_PASS
    if critical_warnings:
        status = STATUS_FAIL
    elif warnings:
        status = STATUS_WARNING

    return {
        "status": status,
        "fields": field_results,
        "warnings": warnings + critical_warnings,
        "critical_warnings": critical_warnings,
    }


def validate_pipeline(
    text: str,
    metadata: dict[str, Any] | None,
    document_type: str,
    classification_confidence: float | None,
    confidence_source: str | None,
    fields: dict[str, Any],
) -> dict[str, Any]:
    """Run all advisory validation checks and summarize pipeline reliability."""
    ocr_quality = validate_text_quality(text, metadata)
    classification = validate_classification(
        document_type,
        classification_confidence,
        confidence_source,
    )
    field_validation = validate_fields(document_type, fields)

    components = (ocr_quality, classification, field_validation)
    warnings = _combined_warnings(components)
    critical_warnings = _combined_critical_warnings(components)
    total_warnings = len(warnings)
    critical_warning_count = len(critical_warnings)
    pipeline_status = _pipeline_status(components)
    validation_score = _validation_score(total_warnings, critical_warning_count)

    return {
        "ocr_quality": ocr_quality,
        "classification": classification,
        "fields": field_validation,
        "pipeline_status": pipeline_status,
        "total_warnings": total_warnings,
        "critical_warning_count": critical_warning_count,
        "validation_score": validation_score,
        "warnings": warnings,
    }


def _validate_field(field_name: str, value: Any, document_type: str) -> dict[str, Any]:
    warnings: list[str] = []
    critical_warnings: list[str] = []
    valid = True

    if value in (None, ""):
        if field_name == "invoice_number" and document_type != "receipt":
            warnings.append(f"{field_name} is missing.")
        elif field_name in {"date", "amount", "supplier"}:
            warnings.append(f"{field_name} is missing.")
        return _field_result(value, valid, warnings, critical_warnings)

    text_value = str(value).strip()
    if field_name == "date" and not _valid_date(text_value):
        valid = False
        critical_warnings.append(f"date has an unsupported format: {text_value}")
    elif field_name == "amount" and not _valid_amount(text_value):
        valid = False
        critical_warnings.append(f"amount does not look numeric: {text_value}")
    elif field_name == "supplier" and _bad_supplier(text_value):
        valid = False
        critical_warnings.append(f"supplier looks like a label/noise value: {text_value}")
    elif field_name == "invoice_number" and _bad_document_number(text_value):
        valid = False
        critical_warnings.append(f"document number looks invalid: {text_value}")

    return _field_result(value, valid, warnings, critical_warnings)


def _field_result(
    value: Any,
    valid: bool,
    warnings: list[str],
    critical_warnings: list[str],
) -> dict[str, Any]:
    return {
        "value": value,
        "valid": valid,
        "status": STATUS_FAIL if critical_warnings else STATUS_WARNING if warnings else STATUS_PASS,
        "warnings": warnings + critical_warnings,
        "critical_warnings": critical_warnings,
    }


def _valid_date(value: str) -> bool:
    formats = (
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d.%m.%Y",
        "%m.%d.%Y",
        "%Y-%m-%d",
        "%B %d %Y",
        "%b %d %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d-%b-%Y",
        "%d-%B-%Y",
    )
    for date_format in formats:
        try:
            parsed = datetime.strptime(value, date_format)
            return 1990 <= parsed.year <= 2100
        except ValueError:
            continue
    return False


def _valid_amount(value: str) -> bool:
    cleaned = value.replace(",", "").strip()
    return re.fullmatch(r"(?:Rs\.?|RM|\$)?\s*\d+(?:\.\d{1,2})?", cleaned, re.IGNORECASE) is not None


def _bad_supplier(value: str) -> bool:
    cleaned = " ".join(value.lower().split()).strip(" .,:;-")
    if cleaned in BAD_SUPPLIER_VALUES:
        return True
    return any(marker in cleaned for marker in BAD_SUPPLIER_VALUES)


def _bad_document_number(value: str) -> bool:
    return len(value.strip()) < 3 or not re.search(r"\d", value)


def _normalize_document_type(document_type: str) -> str:
    return str(document_type or "").strip().lower().replace(" ", "_").replace("-", "_")


def _is_ocr_method(extraction_method: Any) -> bool:
    return "ocr" in str(extraction_method or "").lower() or "paddle" in str(extraction_method or "").lower()


def _combined_warnings(components: tuple[dict[str, Any], ...]) -> list[str]:
    warnings: list[str] = []
    for component in components:
        warnings.extend(str(warning) for warning in component.get("warnings", []))
    return warnings


def _combined_critical_warnings(components: tuple[dict[str, Any], ...]) -> list[str]:
    warnings: list[str] = []
    for component in components:
        warnings.extend(str(warning) for warning in component.get("critical_warnings", []))
    return warnings


def _pipeline_status(components: tuple[dict[str, Any], ...]) -> str:
    if any(component.get("status") == STATUS_FAIL for component in components):
        return PIPELINE_REVIEW
    if any(component.get("status") == STATUS_WARNING for component in components):
        return PIPELINE_WARNINGS
    return PIPELINE_PROCESSED


def _validation_score(total_warnings: int, critical_warning_count: int) -> float:
    score = 1.0 - (total_warnings * 0.08) - (critical_warning_count * 0.25)
    return round(max(0.0, min(1.0, score)), 2)


if __name__ == "__main__":
    clean_invoice = validate_pipeline(
        text="SuperStore\nInvoice No 39519\nDate 2026-05-05\nTotal Amount $1,250.00",
        metadata={"extraction_method": "pymupdf"},
        document_type="invoice",
        classification_confidence=None,
        confidence_source="heuristic",
        fields={
            "invoice_number": "39519",
            "date": "2026-05-05",
            "amount": "$1,250.00",
            "supplier": "SuperStore",
        },
    )
    noisy_ocr = validate_pipeline(
        text="@@@ ### ??",
        metadata={"extraction_method": "paddleocr_image"},
        document_type="receipt",
        classification_confidence=0.80,
        confidence_source="model",
        fields={"invoice_number": None, "date": None, "amount": None, "supplier": None},
    )
    low_confidence = validate_classification("invoice", 0.32, "model")
    invalid_supplier = validate_fields(
        "invoice",
        {
            "invoice_number": "39519",
            "date": "2026-05-05",
            "amount": "$22.17",
            "supplier": "Bill To",
        },
    )
    print("clean_invoice:", clean_invoice["pipeline_status"], clean_invoice["validation_score"])
    print("noisy_ocr:", noisy_ocr["pipeline_status"], noisy_ocr["validation_score"])
    print("low_confidence:", low_confidence["status"], low_confidence["warnings"])
    print("invalid_supplier:", invalid_supplier["status"], invalid_supplier["warnings"])
