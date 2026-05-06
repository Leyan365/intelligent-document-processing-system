"""Shared helpers for lightweight local evaluation scripts."""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any


FIELD_NAMES = ("invoice_number", "date", "amount", "supplier")


def normalize_text(text: Any) -> str:
    """Normalize text for approximate value matching."""
    if text is None:
        return ""
    normalized = str(text).lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    return normalized.strip()


def safe_string_match(a: Any, b: Any) -> bool:
    """Return True for conservative exact-or-contained normalized matches."""
    left = normalize_text(a)
    right = normalize_text(b)
    if not left or not right:
        return False
    return left == right or left in right or right in left


def field_from_question(text: str) -> str | None:
    """Best-effort FUNSD question/label to IDP field mapping."""
    normalized = normalize_text(text)
    if not normalized:
        return None

    if any(token in normalized for token in ("invoice", " id", " no", "number")):
        return "invoice_number"
    if "date" in normalized:
        return "date"
    if any(token in normalized for token in ("total", "amount", "balance", "price")) or "$" in text:
        return "amount"
    if any(token in normalized for token in ("company", "vendor", "from", "supplier")):
        return "supplier"
    return None


def extract_funsd_ground_truth(annotation_path: Path) -> dict[str, set[str]]:
    """Extract approximate target fields from FUNSD annotations."""
    with annotation_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    items = data.get("form", [])
    by_id = {item.get("id"): item for item in items}
    ground_truth: dict[str, set[str]] = {field: set() for field in FIELD_NAMES}

    for item in items:
        if item.get("label") != "question":
            continue

        field = field_from_question(item.get("text", ""))
        if field is None:
            continue

        for source_id, target_id in item.get("linking", []):
            answer_id = target_id if source_id == item.get("id") else source_id
            answer = by_id.get(answer_id)
            if answer and answer.get("label") == "answer":
                value = str(answer.get("text", "")).strip()
                if value:
                    ground_truth[field].add(value)

    return ground_truth


def precision_recall_f1(true_positive: int, false_positive: int, false_negative: int) -> tuple[float, float, float]:
    """Compute precision, recall, and f1 with zero-safe division."""
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return precision, recall, f1


def find_matching_image(images_dir: Path, annotation_path: Path) -> Path | None:
    """Find a FUNSD image by annotation stem without scanning the image tree."""
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = images_dir / f"{annotation_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def can_read_image(image_path: Path) -> tuple[bool, str | None]:
    """Check image readability with OpenCV first, then PIL fallback."""
    cv2_error = None
    try:
        import cv2

        image = cv2.imread(str(image_path))
        if image is not None:
            return True, None
        cv2_error = "cv2.imread returned None"
    except Exception as exc:
        cv2_error = f"cv2: {type(exc).__name__}: {exc}"

    try:
        from PIL import Image

        with Image.open(image_path) as image:
            image.verify()
        return True, None
    except Exception as exc:
        return False, f"{cv2_error}; PIL: {type(exc).__name__}: {exc}"


def add_example_error(errors: list[str], path: Path, error: str, limit: int = 3) -> None:
    """Keep up to a few short example errors for reporting."""
    if len(errors) < limit:
        errors.append(f"{path.name}: {error[:180]}")
