"""Invoice detection sanity check on the local RVL-CDIP subset."""

from __future__ import annotations

import argparse
import random
import sys
import traceback
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from idp_system.pipeline.classifier import DocumentClassifier
from idp_system.pipeline.loader import extract_text
from idp_system.pipeline.ocr import OCRService

from utils import add_example_error, can_read_image


DATA_ROOT = PROJECT_ROOT / "data" / "rvl_cdip"
MAX_SAMPLES = 100
PROGRESS_EVERY = 10
RANDOM_SEED = 42


def collect_invoice_paths() -> list[Path]:
    """List invoice image files once from train and val invoice folders."""
    paths: list[Path] = []
    for split in ("train", "val"):
        invoice_dir = DATA_ROOT / split / "invoice"
        if invoice_dir.exists():
            paths.extend(sorted(invoice_dir.glob("*.tif")))
    return paths


def evaluate_invoice_detection(
    max_samples: int = MAX_SAMPLES,
    progress_every: int = PROGRESS_EVERY,
) -> dict[str, float | int]:
    random.seed(RANDOM_SEED)
    all_paths = collect_invoice_paths()
    sampled_paths = random.sample(all_paths, min(max_samples, len(all_paths)))

    classifier = DocumentClassifier()
    ocr_service = OCRService()
    predicted_as_invoice = 0
    predicted_as_other = 0
    skipped_unreadable = 0
    skipped_extraction_failed = 0
    skipped_empty_text = 0
    example_errors: list[str] = []
    tracebacks: list[str] = []

    verbose_per_file = max_samples <= 5

    for index, image_path in enumerate(sampled_paths, start=1):
        file_start = perf_counter()
        should_log_progress = (
            verbose_per_file
            or index == 1
            or index % progress_every == 0
            or index == len(sampled_paths)
        )
        if should_log_progress:
            print(
                f"Processing {index}/{len(sampled_paths)} sampled RVL-CDIP files: "
                f"{image_path.name}"
            )

        readable, read_error = can_read_image(image_path)
        if not readable:
            skipped_unreadable += 1
            add_example_error(example_errors, image_path, read_error or "unreadable image")
            if verbose_per_file:
                print(f"Finished {image_path.name} in {perf_counter() - file_start:.2f}s.")
            continue

        try:
            extraction = extract_text(image_path, ocr_service=ocr_service)
        except Exception as exc:
            skipped_extraction_failed += 1
            add_example_error(example_errors, image_path, f"{type(exc).__name__}: {exc}")
            if len(tracebacks) < 3:
                tracebacks.append(f"{image_path.name}\n{traceback.format_exc()}")
            if verbose_per_file:
                print(f"Finished {image_path.name} in {perf_counter() - file_start:.2f}s.")
            continue

        if not extraction.text.strip():
            skipped_empty_text += 1
            add_example_error(example_errors, image_path, "empty extracted text")
            if verbose_per_file:
                print(f"Finished {image_path.name} in {perf_counter() - file_start:.2f}s.")
            continue

        prediction = classifier.classify(extraction.text)

        if prediction == "invoice":
            predicted_as_invoice += 1
        else:
            predicted_as_other += 1

        if verbose_per_file:
            print(f"Finished {image_path.name} in {perf_counter() - file_start:.2f}s.")

    total_samples = predicted_as_invoice + predicted_as_other
    accuracy = predicted_as_invoice / total_samples if total_samples else 0.0

    return {
        "sampled_files": len(sampled_paths),
        "processed": total_samples,
        "skipped_unreadable": skipped_unreadable,
        "skipped_extraction_failed": skipped_extraction_failed,
        "skipped_empty_text": skipped_empty_text,
        "total_samples": total_samples,
        "predicted_as_invoice": predicted_as_invoice,
        "predicted_as_other": predicted_as_other,
        "accuracy": accuracy,
        "example_errors": example_errors,
        "tracebacks": tracebacks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Invoice detection sanity check on the local RVL-CDIP subset."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=MAX_SAMPLES,
        help=f"Maximum RVL-CDIP invoice files to sample. Default: {MAX_SAMPLES}.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=PROGRESS_EVERY,
        help=f"Print progress before every Nth file. Default: {PROGRESS_EVERY}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progress_every = max(1, args.progress_every)
    max_samples = max(0, args.max_samples)
    metrics = evaluate_invoice_detection(
        max_samples=max_samples,
        progress_every=progress_every,
    )
    print("Classification Evaluation (RVL-CDIP):")
    print("This is not a full classification evaluation. Only invoice samples were available locally. Multi-class evaluation requires additional labeled data.")
    print(f"- sampled_files: {metrics['sampled_files']}")
    print(f"- processed: {metrics['processed']}")
    print(f"- skipped_unreadable: {metrics['skipped_unreadable']}")
    print(f"- skipped_extraction_failed: {metrics['skipped_extraction_failed']}")
    print(f"- skipped_empty_text: {metrics['skipped_empty_text']}")
    print(f"- total_samples: {metrics['total_samples']}")
    print(f"- predicted_as_invoice: {metrics['predicted_as_invoice']}")
    print(f"- predicted_as_other: {metrics['predicted_as_other']}")
    print(f"- accuracy: {metrics['accuracy']:.2f}")
    if metrics["example_errors"]:
        print("- example_errors:")
        for error in metrics["example_errors"]:
            print(f"  - {error}")
    if metrics["tracebacks"]:
        print("- tracebacks:")
        for item in metrics["tracebacks"]:
            print(item)


if __name__ == "__main__":
    main()
