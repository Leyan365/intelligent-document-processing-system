"""Approximate extraction evaluation on small FUNSD samples."""

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
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from idp_system.pipeline.extractor import InformationExtractor
from idp_system.pipeline.loader import extract_text
from idp_system.pipeline.ocr import OCRService

from utils import (
    FIELD_NAMES,
    add_example_error,
    can_read_image,
    extract_funsd_ground_truth,
    find_matching_image,
    precision_recall_f1,
    safe_string_match,
)


DATA_ROOT = PROJECT_ROOT / "data" / "funsd"
MAX_DOCUMENTS = 50
PROGRESS_EVERY = 10
RANDOM_SEED = 42


def collect_annotation_paths() -> list[Path]:
    """List FUNSD annotation files once from training and testing splits."""
    paths: list[Path] = []
    for split in ("training_data", "testing_data"):
        annotations_dir = DATA_ROOT / split / "annotations"
        if annotations_dir.exists():
            paths.extend(sorted(annotations_dir.glob("*.json")))
    return paths


def image_dir_for_annotation(annotation_path: Path) -> Path:
    return annotation_path.parents[1] / "images"


def evaluate_extraction(
    max_documents: int = MAX_DOCUMENTS,
    progress_every: int = PROGRESS_EVERY,
) -> dict[str, object]:
    random.seed(RANDOM_SEED)
    all_annotations = collect_annotation_paths()
    sampled_annotations = random.sample(all_annotations, min(max_documents, len(all_annotations)))

    extractor = InformationExtractor()
    ocr_service = OCRService()
    counts = {field: {"tp": 0, "fp": 0, "fn": 0} for field in FIELD_NAMES}
    skipped_unreadable = 0
    skipped_extraction_failed = 0
    skipped_empty_text = 0
    processed_documents = 0
    documents_with_any_match = 0
    example_errors: list[str] = []
    tracebacks: list[str] = []

    verbose_per_file = max_documents <= 5

    for index, annotation_path in enumerate(sampled_annotations, start=1):
        file_start = perf_counter()
        should_log_progress = (
            verbose_per_file
            or index == 1
            or index % progress_every == 0
            or index == len(sampled_annotations)
        )
        if should_log_progress:
            print(
                f"Processing {index}/{len(sampled_annotations)} sampled FUNSD documents: "
                f"{annotation_path.name}"
            )

        image_path = find_matching_image(image_dir_for_annotation(annotation_path), annotation_path)
        if image_path is None:
            skipped_unreadable += 1
            add_example_error(example_errors, annotation_path, "matching image not found")
            if verbose_per_file:
                print(f"Finished {annotation_path.name} in {perf_counter() - file_start:.2f}s.")
            continue

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

        try:
            predicted = extractor.extract(extraction.text)
            ground_truth = extract_funsd_ground_truth(annotation_path)
        except Exception as exc:
            skipped_extraction_failed += 1
            add_example_error(example_errors, annotation_path, f"{type(exc).__name__}: {exc}")
            if len(tracebacks) < 3:
                tracebacks.append(f"{annotation_path.name}\n{traceback.format_exc()}")
            if verbose_per_file:
                print(f"Finished {annotation_path.name} in {perf_counter() - file_start:.2f}s.")
            continue

        processed_documents += 1
        document_matched = False

        for field in FIELD_NAMES:
            prediction = predicted.get(field)
            targets = ground_truth[field]
            matched = prediction is not None and any(safe_string_match(prediction, target) for target in targets)

            if matched:
                counts[field]["tp"] += 1
                document_matched = True
            elif prediction:
                counts[field]["fp"] += 1

            if targets and not matched:
                counts[field]["fn"] += 1

        if document_matched:
            documents_with_any_match += 1

        if verbose_per_file:
            print(f"Finished {image_path.name} in {perf_counter() - file_start:.2f}s.")

    metrics = {}
    for field, field_counts in counts.items():
        precision, recall, f1 = precision_recall_f1(
            field_counts["tp"],
            field_counts["fp"],
            field_counts["fn"],
        )
        metrics[field] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "sampled_documents": len(sampled_annotations),
        "processed": processed_documents,
        "skipped_unreadable": skipped_unreadable,
        "skipped_extraction_failed": skipped_extraction_failed,
        "skipped_empty_text": skipped_empty_text,
        "total_documents": processed_documents,
        "documents_with_any_match": documents_with_any_match,
        "fields": metrics,
        "example_errors": example_errors,
        "tracebacks": tracebacks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approximate extraction evaluation on small FUNSD samples."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=MAX_DOCUMENTS,
        help=f"Maximum FUNSD documents to sample. Default: {MAX_DOCUMENTS}.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=PROGRESS_EVERY,
        help=f"Print progress before every Nth document. Default: {PROGRESS_EVERY}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progress_every = max(1, args.progress_every)
    max_documents = max(0, args.max_samples)
    results = evaluate_extraction(
        max_documents=max_documents,
        progress_every=progress_every,
    )
    print("Extraction Evaluation (FUNSD):")
    print("FUNSD does not directly map to invoice/PO fields, so results are approximate.")
    print(f"- sampled_documents: {results['sampled_documents']}")
    print(f"- processed: {results['processed']}")
    print(f"- skipped_unreadable: {results['skipped_unreadable']}")
    print(f"- skipped_extraction_failed: {results['skipped_extraction_failed']}")
    print(f"- skipped_empty_text: {results['skipped_empty_text']}")
    print(f"- total_documents: {results['total_documents']}")
    print(f"- documents_with_any_match: {results['documents_with_any_match']}")
    if results["example_errors"]:
        print("- example_errors:")
        for error in results["example_errors"]:
            print(f"  - {error}")
    if results["tracebacks"]:
        print("- tracebacks:")
        for item in results["tracebacks"]:
            print(item)
    print()

    fields = results["fields"]
    for field in FIELD_NAMES:
        metrics = fields[field]
        print(f"{field}:")
        print(f"  precision: {metrics['precision']:.2f}")
        print(f"  recall: {metrics['recall']:.2f}")
        print(f"  f1: {metrics['f1']:.2f}")
        print()


if __name__ == "__main__":
    main()
