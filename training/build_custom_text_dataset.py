"""Build a local 3-class text dataset for document classification."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = DATA_ROOT / "custom_text_dataset"

RVL_CACHE_ROOT = DATA_ROOT / "processed" / "rvl_text_cache"
PO_TEXT_ROOT = DATA_ROOT / "custom_po_text"
SROIE_ROOT = DATA_ROOT / "sroie"

SPLITS = ("train", "val")
LABELS = ("invoice", "purchase_order", "receipt")


def normalize_text(text: str) -> str:
    """Apply light whitespace normalization only."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def read_source_text(path: Path) -> str:
    """Read dataset source text with a small set of local-file encoding fallbacks."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def read_plain_text(path: Path) -> str:
    return normalize_text(read_source_text(path))


def parse_sroie_box_text(path: Path) -> str:
    """Extract readable OCR text from SROIE box metadata lines."""
    lines: list[str] = []
    for raw_line in read_source_text(path).splitlines():
        text = extract_sroie_line_text(raw_line)
        if text:
            lines.append(text)
    return normalize_text("\n".join(lines))


def extract_sroie_line_text(line: str) -> str:
    """Remove leading coordinate fields and keep the OCR text content."""
    parts = [part.strip() for part in line.split(",")]
    if not parts:
        return ""

    if len(parts) > 8 and all(is_number(part) for part in parts[:8]):
        text_parts = parts[8:]
    elif len(parts) > 4 and all(is_number(part) for part in parts[:4]):
        text_parts = parts[4:]
    else:
        text_parts = parts

    if len(text_parts) > 1 and is_confidence(text_parts[-1]):
        text_parts = text_parts[:-1]

    return ",".join(text_parts).strip()


def is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def is_confidence(value: str) -> bool:
    try:
        number = float(value)
    except ValueError:
        return False
    return 0.0 <= number <= 1.0


def ensure_output_dirs() -> None:
    for split in SPLITS:
        for label in LABELS:
            (OUTPUT_ROOT / split / label).mkdir(parents=True, exist_ok=True)


def write_dataset_split(
    source_dir: Path,
    output_dir: Path,
    prefix: str,
    reader: Callable[[Path], str],
) -> tuple[int, int]:
    """Write normalized source text files into one dataset split/label folder."""
    count = 0
    failed = 0
    source_files = sorted(source_dir.glob("*.txt")) if source_dir.exists() else []

    if not source_dir.exists():
        print(f"Skipping missing source folder: {source_dir}")

    for source_path in source_files:
        try:
            text = reader(source_path)
            if not text:
                continue

            count += 1
            output_path = output_dir / f"{prefix}_{count:04d}.txt"
            output_path.write_text(text, encoding="utf-8")
        except Exception as exc:
            failed += 1
            print(f"Failed {source_path}: {type(exc).__name__}: {exc}")

    return count, failed


def build_dataset() -> dict[str, int]:
    ensure_output_dirs()
    summary: dict[str, int] = {}
    failed = 0

    for split in SPLITS:
        count, split_failed = write_dataset_split(
            RVL_CACHE_ROOT / split / "invoice",
            OUTPUT_ROOT / split / "invoice",
            "invoice",
            read_plain_text,
        )
        summary[f"invoice_{split}"] = count
        failed += split_failed

    for split in SPLITS:
        count, split_failed = write_dataset_split(
            PO_TEXT_ROOT / split,
            OUTPUT_ROOT / split / "purchase_order",
            "po",
            read_plain_text,
        )
        summary[f"purchase_order_{split}"] = count
        failed += split_failed

    receipt_sources = {
        "train": SROIE_ROOT / "train" / "box",
        "val": SROIE_ROOT / "test" / "box",
    }
    for split, source_dir in receipt_sources.items():
        count, split_failed = write_dataset_split(
            source_dir,
            OUTPUT_ROOT / split / "receipt",
            "receipt",
            parse_sroie_box_text,
        )
        summary[f"receipt_{split}"] = count
        failed += split_failed

    summary["failed"] = failed
    return summary


def print_sample_output() -> None:
    sample_paths = (
        OUTPUT_ROOT / "train" / "invoice" / "invoice_0001.txt",
        OUTPUT_ROOT / "train" / "purchase_order" / "po_0001.txt",
        OUTPUT_ROOT / "train" / "receipt" / "receipt_0001.txt",
    )
    for sample_path in sample_paths:
        if not sample_path.exists():
            continue
        sample = sample_path.read_text(encoding="utf-8")[:300].strip()
        print(f"\nSample {sample_path.relative_to(PROJECT_ROOT)}:")
        print(sample)


def main() -> None:
    summary = build_dataset()
    print("Custom Text Dataset Summary:")
    print(f"- invoice_train: {summary['invoice_train']}")
    print(f"- invoice_val: {summary['invoice_val']}")
    print(f"- purchase_order_train: {summary['purchase_order_train']}")
    print(f"- purchase_order_val: {summary['purchase_order_val']}")
    print(f"- receipt_train: {summary['receipt_train']}")
    print(f"- receipt_val: {summary['receipt_val']}")
    print(f"- failed: {summary['failed']}")
    print_sample_output()


if __name__ == "__main__":
    main()
