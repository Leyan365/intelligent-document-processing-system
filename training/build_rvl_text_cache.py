"""Build cached OCR text files for RVL-CDIP invoice samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from idp_system.pipeline.loader import extract_text
from idp_system.pipeline.ocr import OCRService


DATA_ROOT = PROJECT_ROOT / "data" / "rvl_cdip"
CACHE_ROOT = PROJECT_ROOT / "data" / "processed" / "rvl_text_cache"
VALID_SPLITS = ("train", "val")
DEFAULT_PROGRESS_EVERY = 10


def collect_invoice_paths(split: str) -> list[Path]:
    """Collect invoice TIFF paths for a single RVL-CDIP split."""
    invoice_dir = DATA_ROOT / split / "invoice"
    if not invoice_dir.exists():
        return []
    return sorted(invoice_dir.glob("*.tif"))


def cache_path_for(image_path: Path, split: str) -> Path:
    """Map an RVL-CDIP image path to its text cache path."""
    return CACHE_ROOT / split / "invoice" / f"{image_path.stem}.txt"


def selected_splits(split: str) -> tuple[str, ...]:
    if split == "all":
        return VALID_SPLITS
    return (split,)


def build_cache(
    split: str,
    max_samples: int | None,
    progress_every: int,
    overwrite: bool,
) -> dict[str, int]:
    """Build OCR text cache files using one shared OCRService instance."""
    paths_by_split = {
        current_split: collect_invoice_paths(current_split)
        for current_split in selected_splits(split)
    }
    selected_paths: list[tuple[str, Path]] = []
    for current_split, paths in paths_by_split.items():
        for image_path in paths:
            selected_paths.append((current_split, image_path))

    if max_samples is not None:
        selected_paths = selected_paths[:max_samples]

    ocr_service = OCRService()
    processed = 0
    skipped_existing = 0
    failed = 0
    empty_text = 0
    progress_every = max(1, progress_every)
    total = len(selected_paths)

    for index, (current_split, image_path) in enumerate(selected_paths, start=1):
        output_path = cache_path_for(image_path, current_split)
        print(f"Processing {index}/{total}: {current_split}/invoice/{image_path.name}")

        if output_path.exists() and not overwrite:
            skipped_existing += 1
            print(f"Skipped existing cache: {output_path}")
            if index % progress_every == 0 or index == total:
                print(f"Progress: {index}/{total} files checked.")
            continue

        start_time = perf_counter()
        try:
            extraction = extract_text(image_path, ocr_service=ocr_service)
            text = extraction.text.strip()
            if not text:
                empty_text += 1
                print(f"Empty OCR text: {image_path.name} ({perf_counter() - start_time:.2f}s)")
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")
            processed += 1
            print(f"Cached {output_path} ({perf_counter() - start_time:.2f}s)")
        except Exception as exc:
            failed += 1
            print(
                f"Failed {current_split}/invoice/{image_path.name} "
                f"after {perf_counter() - start_time:.2f}s: {type(exc).__name__}: {exc}"
            )

        if index % progress_every == 0 or index == total:
            print(f"Progress: {index}/{total} files checked.")

    return {
        "selected": total,
        "processed": processed,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "empty_text": empty_text,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cached OCR text files for RVL-CDIP invoice samples."
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "all"),
        default="all",
        help="RVL-CDIP split to process. Default: all.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of invoice images to process across selected splits.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help=f"Print progress before every Nth file. Default: {DEFAULT_PROGRESS_EVERY}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate text files even when a cache file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_samples = None if args.max_samples is None else max(0, args.max_samples)
    metrics = build_cache(
        split=args.split,
        max_samples=max_samples,
        progress_every=args.progress_every,
        overwrite=args.overwrite,
    )

    print("RVL-CDIP Invoice OCR Text Cache:")
    print(f"- selected: {metrics['selected']}")
    print(f"- processed: {metrics['processed']}")
    print(f"- skipped_existing: {metrics['skipped_existing']}")
    print(f"- failed: {metrics['failed']}")
    print(f"- empty_text: {metrics['empty_text']}")


if __name__ == "__main__":
    main()
