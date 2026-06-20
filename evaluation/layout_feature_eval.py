"""Compare text-only and layout-proxy document classifiers.

Phase 14 is a research/evaluation experiment. It does not modify or save the
production classifier. The goal is to compare the current text-only TF-IDF
approach with a CPU-friendly model that adds lightweight layout-inspired
numeric features extracted from OCR/text structure.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


LABELS = ("invoice", "purchase_order", "receipt")
RANDOM_SEED = 42
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "custom_text_dataset" / "train"
DEFAULT_VAL_DIR = PROJECT_ROOT / "data" / "custom_text_dataset" / "val"

AMOUNT_PATTERN = re.compile(
    r"(?:Rs\.?|RM|\$)?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})|"
    r"(?:Rs\.?|RM|\$)?\s*\d+(?:\.\d{2})",
    re.IGNORECASE,
)
DATE_PATTERN = re.compile(
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|"
    r"\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*-?\d{4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|"
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}",
    re.IGNORECASE,
)

LAYOUT_FEATURE_NAMES = (
    "char_count",
    "word_count",
    "line_count",
    "average_line_length",
    "max_line_length",
    "digit_ratio",
    "uppercase_ratio",
    "currency_symbol_count",
    "amount_like_count",
    "date_like_count",
    "top_region_invoice_keyword_count",
    "top_region_receipt_keyword_count",
    "top_region_po_keyword_count",
    "bottom_region_total_keyword_count",
    "supplier_label_count",
    "bill_to_ship_to_count",
    "cashier_keyword_count",
    "change_keyword_count",
    "payment_keyword_count",
    "item_table_keyword_count",
    "purchase_order_keyword_count",
    "receipt_keyword_count",
    "invoice_keyword_count",
    "line_item_density_proxy",
    "total_near_bottom_proxy",
)


@dataclass(slots=True)
class DatasetSplit:
    """Loaded texts, labels, and counts for one dataset split."""

    texts: list[str]
    labels: list[str]
    counts: Counter[str]


@dataclass(slots=True)
class ModelMetrics:
    """Evaluation metrics for one classifier."""

    name: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    report: str
    matrix: np.ndarray
    predictions: np.ndarray


@dataclass(slots=True)
class LayoutModelBundle:
    """Fitted layout-aware model pieces needed for interpretation."""

    vectorizer: TfidfVectorizer
    scaler: StandardScaler
    classifier: LogisticRegression


def load_split(split_dir: Path, max_per_class: int | None = None) -> DatasetSplit:
    """Load one text dataset split from class folders."""
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {split_dir}. "
            "Build data/custom_text_dataset before running Phase 14."
        )

    texts: list[str] = []
    labels: list[str] = []
    counts: Counter[str] = Counter()
    rng = random.Random(RANDOM_SEED)
    missing_classes: list[str] = []

    for label in LABELS:
        class_dir = split_dir / label
        if not class_dir.exists():
            missing_classes.append(label)
            continue

        files = sorted(class_dir.glob("*.txt"))
        if max_per_class is not None and len(files) > max_per_class:
            files = sorted(rng.sample(files, max_per_class))

        for path in files:
            text = read_text(path)
            if not text:
                continue
            texts.append(text)
            labels.append(label)
            counts[label] += 1

    if missing_classes:
        raise FileNotFoundError(
            f"Missing class folders under {split_dir}: {', '.join(missing_classes)}"
        )
    if not texts:
        raise ValueError(f"No text files found in dataset split: {split_dir}")
    return DatasetSplit(texts=texts, labels=labels, counts=counts)


def read_text(path: Path) -> str:
    """Read a local text file with simple encoding fallbacks."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace").strip()


def extract_layout_features(text: str) -> list[float]:
    """Extract layout-proxy numeric features from text structure."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_lengths = [len(line) for line in lines]
    line_count = len(lines)
    char_count = len(text)
    words = re.findall(r"\b[\w'-]+\b", text)
    word_count = len(words)
    non_space_chars = [char for char in text if not char.isspace()]
    base_count = len(non_space_chars) or 1
    digit_ratio = sum(char.isdigit() for char in non_space_chars) / base_count
    uppercase_ratio = sum(char.isupper() for char in non_space_chars) / base_count

    top_region = region_text(text, lines, top=True)
    bottom_region = region_text(text, lines, top=False)
    lowered_text = text.lower()
    lowered_top = top_region.lower()
    lowered_bottom = bottom_region.lower()

    table_keyword_count = keyword_count(
        lowered_text,
        ("qty", "quantity", "unit price", "uom", "item", "description", "sku"),
    )
    line_item_density = table_keyword_count / max(line_count, 1)

    return [
        float(char_count),
        float(word_count),
        float(line_count),
        float(sum(line_lengths) / line_count if line_count else 0.0),
        float(max(line_lengths) if line_lengths else 0),
        digit_ratio,
        uppercase_ratio,
        float(sum(text.count(symbol) for symbol in ("$", "Rs", "RM"))),
        float(len(AMOUNT_PATTERN.findall(text))),
        float(len(DATE_PATTERN.findall(text))),
        float(keyword_count(lowered_top, ("invoice", "invoice no", "invoice number", "inv"))),
        float(keyword_count(lowered_top, ("receipt", "cashier", "tender", "change"))),
        float(keyword_count(lowered_top, ("purchase order", "po number", "order number", "supplier"))),
        float(keyword_count(lowered_bottom, ("total", "grand total", "balance due", "amount payable"))),
        float(keyword_count(lowered_text, ("supplier", "supplier name", "vendor", "company name"))),
        float(keyword_count(lowered_text, ("bill to", "ship to"))),
        float(keyword_count(lowered_text, ("cashier",))),
        float(keyword_count(lowered_text, ("change",))),
        float(keyword_count(lowered_text, ("payment", "paid", "cash", "card", "tender"))),
        float(table_keyword_count),
        float(keyword_count(lowered_text, ("purchase order", "po number", "order number"))),
        float(keyword_count(lowered_text, ("receipt", "cashier", "tender"))),
        float(keyword_count(lowered_text, ("invoice", "invoice no", "invoice number", "balance due"))),
        float(line_item_density),
        float(1.0 if keyword_count(lowered_bottom, ("total", "grand total", "balance due", "amount payable")) else 0.0),
    ]


def region_text(text: str, lines: list[str], top: bool) -> str:
    """Return top/bottom region as 25% of lines plus a 500-character fallback."""
    if lines:
        region_size = max(1, int(np.ceil(len(lines) * 0.25)))
        selected = lines[:region_size] if top else lines[-region_size:]
        line_region = "\n".join(selected)
    else:
        line_region = ""

    char_region = text[:500] if top else text[-500:]
    return f"{line_region}\n{char_region}".strip()


def keyword_count(text: str, keywords: tuple[str, ...]) -> int:
    """Count simple keyword occurrences in normalized text."""
    return sum(text.count(keyword) for keyword in keywords)


def build_layout_matrix(texts: list[str]) -> np.ndarray:
    """Build a dense numeric feature matrix for texts."""
    return np.array([extract_layout_features(text) for text in texts], dtype="float64")


def train_text_only_model(train: DatasetSplit) -> tuple[TfidfVectorizer, LogisticRegression]:
    """Train the text-only TF-IDF + Logistic Regression baseline."""
    vectorizer = build_vectorizer()
    train_matrix = vectorizer.fit_transform(train.texts)
    classifier = build_classifier()
    classifier.fit(train_matrix, train.labels)
    return vectorizer, classifier


def train_layout_model(train: DatasetSplit) -> LayoutModelBundle:
    """Train TF-IDF plus scaled layout-proxy features."""
    vectorizer = build_vectorizer()
    text_matrix = vectorizer.fit_transform(train.texts)
    layout_features = build_layout_matrix(train.texts)
    scaler = StandardScaler()
    scaled_layout = scaler.fit_transform(layout_features)
    train_matrix = hstack([text_matrix, csr_matrix(scaled_layout)], format="csr")
    classifier = build_classifier()
    classifier.fit(train_matrix, train.labels)
    return LayoutModelBundle(vectorizer=vectorizer, scaler=scaler, classifier=classifier)


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=20000,
    )


def build_classifier() -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )


def evaluate_text_model(
    name: str,
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    val: DatasetSplit,
) -> ModelMetrics:
    matrix = vectorizer.transform(val.texts)
    predictions = classifier.predict(matrix)
    return compute_metrics(name, val.labels, predictions)


def evaluate_layout_model(name: str, bundle: LayoutModelBundle, val: DatasetSplit) -> ModelMetrics:
    text_matrix = bundle.vectorizer.transform(val.texts)
    layout_features = build_layout_matrix(val.texts)
    scaled_layout = bundle.scaler.transform(layout_features)
    matrix = hstack([text_matrix, csr_matrix(scaled_layout)], format="csr")
    predictions = bundle.classifier.predict(matrix)
    return compute_metrics(name, val.labels, predictions)


def compute_metrics(name: str, y_true: list[str], predictions: np.ndarray) -> ModelMetrics:
    """Compute standard multiclass classification metrics."""
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=list(LABELS),
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=list(LABELS),
        average="weighted",
        zero_division=0,
    )
    return ModelMetrics(
        name=name,
        accuracy=accuracy_score(y_true, predictions),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        report=classification_report(y_true, predictions, labels=list(LABELS), zero_division=0),
        matrix=confusion_matrix(y_true, predictions, labels=list(LABELS)),
        predictions=predictions,
    )


def top_layout_features_by_class(bundle: LayoutModelBundle, top_n: int = 8) -> dict[str, list[tuple[str, float]]]:
    """Return top numeric layout features by absolute coefficient per class."""
    tfidf_count = len(bundle.vectorizer.get_feature_names_out())
    coefficients = bundle.classifier.coef_[:, tfidf_count:]
    result: dict[str, list[tuple[str, float]]] = {}
    for class_index, class_name in enumerate(bundle.classifier.classes_):
        class_coefficients = coefficients[class_index]
        ordered_indices = np.argsort(np.abs(class_coefficients))[::-1][:top_n]
        result[str(class_name)] = [
            (LAYOUT_FEATURE_NAMES[index], float(class_coefficients[index]))
            for index in ordered_indices
        ]
    return result


def print_counts(train: DatasetSplit, val: DatasetSplit) -> None:
    print("Dataset Counts")
    print("Train:")
    for label in LABELS:
        print(f"- {label}: {train.counts[label]}")
    print("Validation:")
    for label in LABELS:
        print(f"- {label}: {val.counts[label]}")
    print()


def print_metrics(metrics: ModelMetrics) -> None:
    print(metrics.name)
    print(f"- accuracy: {metrics.accuracy:.4f}")
    print(f"- macro_precision: {metrics.macro_precision:.4f}")
    print(f"- macro_recall: {metrics.macro_recall:.4f}")
    print(f"- macro_f1: {metrics.macro_f1:.4f}")
    print(f"- weighted_f1: {metrics.weighted_f1:.4f}")
    print("Classification report:")
    print(metrics.report)
    print("Confusion matrix:")
    print("labels:", ", ".join(LABELS))
    print(metrics.matrix)
    print()


def print_comparison(text_metrics: ModelMetrics, layout_metrics: ModelMetrics) -> None:
    delta_accuracy = layout_metrics.accuracy - text_metrics.accuracy
    delta_macro_f1 = layout_metrics.macro_f1 - text_metrics.macro_f1
    outcome = comparison_outcome(delta_macro_f1)
    print("Model Comparison")
    print("model,accuracy,macro_precision,macro_recall,macro_f1,weighted_f1")
    for metrics in (text_metrics, layout_metrics):
        print(
            f"{metrics.name},{metrics.accuracy:.4f},{metrics.macro_precision:.4f},"
            f"{metrics.macro_recall:.4f},{metrics.macro_f1:.4f},{metrics.weighted_f1:.4f}"
        )
    print(f"- delta_accuracy: {delta_accuracy:+.4f}")
    print(f"- delta_macro_f1: {delta_macro_f1:+.4f}")
    print(f"- layout_feature_effect: {outcome}")
    print()


def comparison_outcome(delta_macro_f1: float) -> str:
    if delta_macro_f1 > 0.0001:
        return "improved"
    if delta_macro_f1 < -0.0001:
        return "reduced"
    return "matched"


def print_top_layout_features(features: dict[str, list[tuple[str, float]]]) -> None:
    print("Top Layout Features By Class")
    for class_name in LABELS:
        print(f"{class_name}:")
        for feature_name, coefficient in features.get(class_name, []):
            print(f"- {feature_name}: {coefficient:+.4f}")
    print()


def print_interpretation(text_metrics: ModelMetrics, layout_metrics: ModelMetrics) -> None:
    effect = comparison_outcome(layout_metrics.macro_f1 - text_metrics.macro_f1)
    print("Interpretation")
    if effect == "improved":
        print(
            "The layout-aware proxy feature model improved macro F1 on the current "
            "validation split, suggesting that lightweight structural cues add useful "
            "classification signal beyond text alone."
        )
    elif effect == "reduced":
        print(
            "The layout-aware proxy feature model reduced macro F1 on the current "
            "validation split. This suggests the present split may already be highly "
            "separable by text/domain cues, while the added layout proxies may introduce "
            "some noise for this dataset."
        )
    else:
        print(
            "The layout-aware feature model matched the text-only baseline on the "
            "current validation split. This may be because the validation split is "
            "already highly separable by text/domain cues. However, layout-aware "
            "features remain important for more diverse multi-layout documents and "
            "are included as a CPU-friendly response to spatial-structure limitations."
        )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare text-only and layout-proxy document classifiers."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help=f"Training split directory. Default: {DEFAULT_TRAIN_DIR}.",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=DEFAULT_VAL_DIR,
        help=f"Validation split directory. Default: {DEFAULT_VAL_DIR}.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional deterministic cap per class per split for smoke runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_per_class = None if args.max_per_class is None else max(1, args.max_per_class)
    train = load_split(args.train_dir, max_per_class)
    val = load_split(args.val_dir, max_per_class)

    print("Layout-Aware Classification Comparison")
    print_counts(train, val)

    text_vectorizer, text_classifier = train_text_only_model(train)
    text_metrics = evaluate_text_model("Text-only TF-IDF + Logistic Regression", text_vectorizer, text_classifier, val)

    layout_bundle = train_layout_model(train)
    layout_metrics = evaluate_layout_model("Text + layout-proxy Logistic Regression", layout_bundle, val)

    print_metrics(text_metrics)
    print_metrics(layout_metrics)
    print_comparison(text_metrics, layout_metrics)
    print_top_layout_features(top_layout_features_by_class(layout_bundle))
    print_interpretation(text_metrics, layout_metrics)


if __name__ == "__main__":
    main()
