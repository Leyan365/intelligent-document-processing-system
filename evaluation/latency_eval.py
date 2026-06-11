"""Benchmark local IDP pipeline stage latency.

Phase 13 measures CPU feasibility by timing each major processing stage with
time.perf_counter(). Only elapsed differences between perf_counter calls are
reported, which makes it suitable for short-duration benchmark measurements.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any


os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from idp_system.core.models import Document, DocumentType
from idp_system.pipeline.classifier import DocumentClassifier
from idp_system.pipeline.extractor import InformationExtractor
from idp_system.pipeline.loader import DocumentLoaderRouter
from idp_system.pipeline.search import SemanticSearchService
from idp_system.pipeline.validation import validate_pipeline
from idp_system.system import _build_search_text


SYNTHETIC_EXTRACTION_METHOD = "not_applicable"
SKIPPED_VALUE = "skipped"
CSV_FIELDS = (
    "document_id",
    "document_name",
    "source",
    "status",
    "error",
    "predicted_type",
    "confidence",
    "confidence_source",
    "extraction_method",
    "text_char_count",
    "validation_pipeline_status",
    "extraction_total_seconds",
    "classification_seconds",
    "extraction_fields_seconds",
    "validation_seconds",
    "embedding_index_seconds",
    "total_seconds",
)


@dataclass(slots=True)
class BenchmarkRow:
    """One benchmark result row for a processed or failed document."""

    document_id: str
    document_name: str
    source: str
    status: str
    error: str = ""
    predicted_type: str = ""
    confidence: float | None = None
    confidence_source: str | None = None
    extraction_method: str = "unknown"
    text_char_count: int = 0
    validation_pipeline_status: str = ""
    extraction_total_seconds: float | None = None
    classification_seconds: float | None = None
    extraction_fields_seconds: float | None = None
    validation_seconds: float | None = None
    embedding_index_seconds: float | None = None
    total_seconds: float | None = None


@dataclass(slots=True)
class PipelineComponents:
    """Shared pipeline components used by the benchmark."""

    loader: DocumentLoaderRouter
    classifier: DocumentClassifier
    extractor: InformationExtractor
    search_service: SemanticSearchService | None
    embedding_index_warning: str | None = None
    indexed_documents: int = 0
    next_synthetic_id: int = 1


@dataclass(slots=True)
class TimedValue:
    """A value returned together with elapsed seconds."""

    value: Any
    seconds: float


def build_synthetic_documents() -> list[Document]:
    """Create small in-memory text documents for dependency-free benchmarking."""
    examples = [
        (
            "synthetic_invoice_superstore",
            "SuperStore invoice",
            (
                "INVOICE # 39519 SuperStore Bill To Aaron Bergman Ship To Main Street. "
                "Invoice Date 2026-05-05. Balance Due $22.17. Total amount due."
            ),
        ),
        (
            "synthetic_receipt_quantum_logic",
            "Quantum Logic Solutions receipt",
            (
                "Company Name: Quantum Logic Solutions RECEIPT. Receipt # 100. "
                "Receipt Date 07/05/2026. TOTAL AMT PAYABLE Rs. 13,500.00. "
                "Paid Amount Rs. 13,500.00. Customer payment cash."
            ),
        ),
        (
            "synthetic_po_screenline",
            "Screenline purchase order",
            (
                "PURCHASE ORDER. Supplier SCREENLINE (PVT) LTD. PO Number 5380034300. "
                "PO Creation Date 25.01.2026. Grand Total 5,746.60. Delivery items."
            ),
        ),
    ]

    return [
        Document(
            id=document_id,
            title=title,
            content=text,
            source="synthetic",
            doc_type=DocumentType.TXT,
            metadata={"synthetic": True},
            extraction_method=SYNTHETIC_EXTRACTION_METHOD,
        )
        for document_id, title, text in examples
    ]


def build_components() -> PipelineComponents:
    """Instantiate components once so per-document timings focus on processing."""
    warning = embedding_index_warning()
    return PipelineComponents(
        loader=DocumentLoaderRouter(),
        classifier=DocumentClassifier(),
        extractor=InformationExtractor(),
        search_service=None if warning else SemanticSearchService(),
        embedding_index_warning=warning,
    )


def embedding_index_warning() -> str | None:
    """Return a clear skip reason when production search dependencies are missing."""
    missing = [
        package
        for package in ("sentence_transformers", "faiss")
        if importlib.util.find_spec(package) is None
    ]
    if not missing:
        return None
    return (
        "embedding/index stage skipped because optional production search "
        f"dependencies are unavailable: {', '.join(missing)}"
    )


def benchmark_synthetic_document(
    document: Document,
    components: PipelineComponents,
) -> BenchmarkRow:
    """Benchmark one in-memory synthetic text document."""
    total_start = perf_counter()
    row = BenchmarkRow(
        document_id=document.id,
        document_name=document.title,
        source=document.source,
        status="processed",
        extraction_method=SYNTHETIC_EXTRACTION_METHOD,
        text_char_count=document.char_count,
        extraction_total_seconds=0.0,
    )
    return process_loaded_document(document, row, components, total_start)


def benchmark_file(path: Path, components: PipelineComponents) -> BenchmarkRow:
    """Benchmark one file using the real document loader/extraction pipeline."""
    total_start = perf_counter()
    row = BenchmarkRow(
        document_id=path.stem,
        document_name=path.name,
        source=str(path),
        status="failed",
    )

    try:
        extraction = timed(lambda: components.loader.load(path))
    except Exception as exc:
        row.error = f"{type(exc).__name__}: {exc}"
        row.extraction_total_seconds = perf_counter() - total_start
        row.total_seconds = row.extraction_total_seconds
        return row

    document = extraction.value
    row.document_id = document.id
    row.document_name = document.title
    row.extraction_method = document.extraction_method or "unknown"
    row.text_char_count = document.char_count
    row.extraction_total_seconds = extraction.seconds
    row.status = "processed"
    return process_loaded_document(document, row, components, total_start)


def process_loaded_document(
    document: Document,
    row: BenchmarkRow,
    components: PipelineComponents,
    total_start: float,
) -> BenchmarkRow:
    """Time classification, field extraction, validation, and search indexing."""
    try:
        classification = timed(lambda: classify_document(components.classifier, document.content))
        row.classification_seconds = classification.seconds
        document_type = str(classification.value["label"])
        row.predicted_type = document_type
        row.confidence = _optional_float(classification.value.get("confidence"))
        row.confidence_source = _optional_string(classification.value.get("confidence_source"))

        field_result = timed(lambda: components.extractor.extract(document.content, document_type))
        row.extraction_fields_seconds = field_result.seconds

        validation_metadata = dict(document.metadata)
        validation_metadata["extraction_method"] = document.extraction_method
        validation_result = timed(
            lambda: validate_pipeline(
                text=document.content,
                metadata=validation_metadata,
                document_type=document_type,
                classification_confidence=row.confidence,
                confidence_source=row.confidence_source,
                fields=field_result.value,
            )
        )
        row.validation_seconds = validation_result.seconds
        row.validation_pipeline_status = str(validation_result.value.get("pipeline_status", "unknown"))

        if components.search_service is not None:
            search_text = _build_search_text(document_type, field_result.value, document.content)
            document_id = document.id or f"synthetic-{components.next_synthetic_id}"
            components.next_synthetic_id += 1
            embedding_result = timed(
                lambda: components.search_service.add_documents(
                    [
                        {
                            "id": document_id,
                            "text": search_text,
                            "type": document_type,
                            "confidence": row.confidence,
                            "confidence_source": row.confidence_source,
                            "fields": field_result.value,
                            "source": document.source,
                        }
                    ]
                )
            )
            row.embedding_index_seconds = embedding_result.seconds
            components.indexed_documents += 1
        else:
            row.embedding_index_seconds = None

        row.status = "processed"
    except Exception as exc:
        row.status = "failed"
        row.error = f"{type(exc).__name__}: {exc}"
    finally:
        row.total_seconds = perf_counter() - total_start

    return row


def classify_document(classifier: DocumentClassifier, text: str) -> dict[str, object]:
    """Mirror IDPSystem classification behavior without importing its helper."""
    if hasattr(classifier, "classify_with_confidence"):
        return classifier.classify_with_confidence(text)
    return {
        "label": classifier.classify(text),
        "confidence": None,
        "confidence_source": None,
    }


def timed(func: Any) -> TimedValue:
    """Run a callable and return its value plus elapsed seconds."""
    start = perf_counter()
    value = func()
    return TimedValue(value=value, seconds=perf_counter() - start)


def aggregate_rows(rows: list[BenchmarkRow]) -> dict[str, object]:
    """Compute summary metrics from successfully processed documents."""
    processed = [row for row in rows if row.status == "processed" and row.total_seconds is not None]
    if not processed:
        return {
            "document_count": 0,
            "average_total_seconds": 0.0,
            "median_total_seconds": 0.0,
            "min_total_seconds": 0.0,
            "max_total_seconds": 0.0,
            "average_stage_seconds": {},
            "slowest_document": None,
            "slowest_stage": None,
        }

    total_times = [float(row.total_seconds) for row in processed]
    average_stage_seconds = {
        "extraction_total_seconds": _mean_optional(row.extraction_total_seconds for row in processed),
        "classification_seconds": _mean_optional(row.classification_seconds for row in processed),
        "extraction_fields_seconds": _mean_optional(row.extraction_fields_seconds for row in processed),
        "validation_seconds": _mean_optional(row.validation_seconds for row in processed),
        "embedding_index_seconds": _mean_optional(row.embedding_index_seconds for row in processed),
    }

    slowest_document = max(processed, key=lambda row: float(row.total_seconds or 0.0))
    slowest_stage = find_slowest_stage(processed)
    return {
        "document_count": len(processed),
        "average_total_seconds": statistics.fmean(total_times),
        "median_total_seconds": statistics.median(total_times),
        "min_total_seconds": min(total_times),
        "max_total_seconds": max(total_times),
        "average_stage_seconds": average_stage_seconds,
        "slowest_document": slowest_document,
        "slowest_stage": slowest_stage,
    }


def find_slowest_stage(rows: list[BenchmarkRow]) -> tuple[str, str, float] | None:
    """Find the largest single measured stage across all processed rows."""
    candidates: list[tuple[str, str, float]] = []
    for row in rows:
        for stage in (
            "extraction_total_seconds",
            "classification_seconds",
            "extraction_fields_seconds",
            "validation_seconds",
            "embedding_index_seconds",
        ):
            value = getattr(row, stage)
            if value is not None:
                candidates.append((row.document_name, stage, float(value)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[2])


def print_report(rows: list[BenchmarkRow], components: PipelineComponents) -> None:
    """Print per-document and aggregate benchmark results."""
    print("CPU Latency Benchmark")
    print(f"- mode_documents: {len(rows)}")
    if components.embedding_index_warning:
        print(f"- warning: {components.embedding_index_warning}")
    print()

    print("Per-document Metrics")
    for row in rows:
        print(f"Document: {row.document_name}")
        print(f"  status: {row.status}")
        if row.error:
            print(f"  error: {row.error}")
        print(f"  predicted_type: {row.predicted_type or 'unknown'}")
        print(f"  extraction_method: {row.extraction_method or 'unknown'}")
        print(f"  text_char_count: {row.text_char_count}")
        print(f"  validation_pipeline_status: {row.validation_pipeline_status or 'unknown'}")
        print(f"  extraction_total_seconds: {_format_seconds(row.extraction_total_seconds)}")
        print(f"  classification_seconds: {_format_seconds(row.classification_seconds)}")
        print(f"  extraction_fields_seconds: {_format_seconds(row.extraction_fields_seconds)}")
        print(f"  validation_seconds: {_format_seconds(row.validation_seconds)}")
        print(f"  embedding_index_seconds: {_format_seconds(row.embedding_index_seconds)}")
        print(f"  total_seconds: {_format_seconds(row.total_seconds)}")
        print()

    aggregates = aggregate_rows(rows)
    print("Aggregate Metrics")
    print(f"- number_of_documents: {aggregates['document_count']}")
    print(f"- average_total_seconds: {aggregates['average_total_seconds']:.6f}")
    print(f"- median_total_seconds: {aggregates['median_total_seconds']:.6f}")
    print(f"- min_total_seconds: {aggregates['min_total_seconds']:.6f}")
    print(f"- max_total_seconds: {aggregates['max_total_seconds']:.6f}")

    print("- average_per_stage_time:")
    stage_averages = aggregates["average_stage_seconds"]
    assert isinstance(stage_averages, dict)
    for stage, value in stage_averages.items():
        print(f"  {stage}: {_format_seconds(value)}")

    slowest_document = aggregates["slowest_document"]
    if isinstance(slowest_document, BenchmarkRow):
        print(
            f"- slowest_document: {slowest_document.document_name} "
            f"({_format_seconds(slowest_document.total_seconds)})"
        )
    else:
        print("- slowest_document: none")

    slowest_stage = aggregates["slowest_stage"]
    if slowest_stage is None:
        print("- slowest_stage_overall: none")
    else:
        document_name, stage, seconds = slowest_stage
        print(f"- slowest_stage_overall: {stage} on {document_name} ({seconds:.6f})")


def write_csv(path: Path, rows: list[BenchmarkRow]) -> None:
    """Write per-document benchmark rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_csv(row))


def row_to_csv(row: BenchmarkRow) -> dict[str, object]:
    """Convert a benchmark row to stable CSV scalar values."""
    return {
        "document_id": row.document_id,
        "document_name": row.document_name,
        "source": row.source,
        "status": row.status,
        "error": row.error,
        "predicted_type": row.predicted_type,
        "confidence": "" if row.confidence is None else row.confidence,
        "confidence_source": row.confidence_source or "",
        "extraction_method": row.extraction_method,
        "text_char_count": row.text_char_count,
        "validation_pipeline_status": row.validation_pipeline_status,
        "extraction_total_seconds": _csv_seconds(row.extraction_total_seconds),
        "classification_seconds": _csv_seconds(row.classification_seconds),
        "extraction_fields_seconds": _csv_seconds(row.extraction_fields_seconds),
        "validation_seconds": _csv_seconds(row.validation_seconds),
        "embedding_index_seconds": _csv_seconds(row.embedding_index_seconds),
        "total_seconds": _csv_seconds(row.total_seconds),
    }


def run_benchmark(files: list[Path] | None) -> tuple[list[BenchmarkRow], PipelineComponents]:
    """Run synthetic or file-mode latency benchmark."""
    components = build_components()
    rows: list[BenchmarkRow] = []
    if files:
        for path in files:
            rows.append(benchmark_file(path, components))
    else:
        for document in build_synthetic_documents():
            rows.append(benchmark_synthetic_document(document, components))
    return rows, components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark stage-level CPU latency for the local IDP pipeline."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=None,
        help="Optional local files to process through the real extraction pipeline.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional CSV path for per-document benchmark rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, components = run_benchmark(args.files)
    print_report(rows, components)
    if args.csv_out:
        write_csv(args.csv_out, rows)
        print()
        print(f"CSV written: {args.csv_out}")


def _format_seconds(value: float | None) -> str:
    if value is None:
        return SKIPPED_VALUE
    return f"{value:.6f}"


def _csv_seconds(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _mean_optional(values: Any) -> float | None:
    materialized = [float(value) for value in values if value is not None]
    if not materialized:
        return None
    return statistics.fmean(materialized)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


if __name__ == "__main__":
    main()
