"""Train the local 3-class document classifier from cached text files."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from idp_system.pipeline.classifier import SUPPORTED_DOCUMENT_TYPES, save_model


DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "custom_text_dataset" / "train"
DEFAULT_VAL_DIR = PROJECT_ROOT / "data" / "custom_text_dataset" / "val"
DEFAULT_MODEL_OUT = PROJECT_ROOT / "models" / "document_classifier.joblib"
LABELS = ("invoice", "purchase_order", "receipt")
RANDOM_SEED = 42


def load_split(split_dir: Path, max_per_class: int | None = None) -> tuple[list[str], list[str], Counter[str]]:
    """Load text files from class folders under a split directory."""
    texts: list[str] = []
    labels: list[str] = []
    counts: Counter[str] = Counter()
    rng = random.Random(RANDOM_SEED)

    for label in LABELS:
        class_dir = split_dir / label
        files = sorted(class_dir.glob("*.txt")) if class_dir.exists() else []
        if max_per_class is not None and len(files) > max_per_class:
            files = sorted(rng.sample(files, max_per_class))

        for path in files:
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            texts.append(text)
            labels.append(label)
            counts[label] += 1

    return texts, labels, counts


def train_model(texts: list[str], labels: list[str]) -> Pipeline:
    """Train TF-IDF + Logistic Regression for document classification."""
    unknown_labels = sorted(set(labels) - set(SUPPORTED_DOCUMENT_TYPES))
    if unknown_labels:
        raise ValueError(f"unsupported labels: {', '.join(unknown_labels)}")

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=20000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    model.fit(texts, labels)
    return model


def print_counts(title: str, counts: Counter[str]) -> None:
    print(title)
    for label in LABELS:
        print(f"- {label}: {counts[label]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the local 3-class document classifier."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help=f"Training dataset directory. Default: {DEFAULT_TRAIN_DIR}.",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=DEFAULT_VAL_DIR,
        help=f"Validation dataset directory. Default: {DEFAULT_VAL_DIR}.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_OUT,
        help=f"Output model path. Default: {DEFAULT_MODEL_OUT}.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap on training samples per class.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_per_class = None if args.max_per_class is None else max(1, args.max_per_class)

    train_texts, train_labels, train_counts = load_split(args.train_dir, max_per_class)
    val_texts, val_labels, val_counts = load_split(args.val_dir)

    if not train_texts:
        raise ValueError(f"No training text files found in {args.train_dir}")
    if not val_texts:
        raise ValueError(f"No validation text files found in {args.val_dir}")

    print_counts("Train sample counts:", train_counts)
    print_counts("Validation sample counts:", val_counts)
    print(f"Train label distribution: {dict(Counter(train_labels))}")
    print(f"Validation label distribution: {dict(Counter(val_labels))}")

    model = train_model(train_texts, train_labels)
    predictions = model.predict(val_texts)

    print(f"Accuracy: {accuracy_score(val_labels, predictions):.4f}")
    print("Classification report:")
    print(classification_report(val_labels, predictions, labels=list(LABELS), zero_division=0))
    print("Confusion matrix:")
    print("labels:", ", ".join(LABELS))
    print(confusion_matrix(val_labels, predictions, labels=list(LABELS)))

    output_path = save_model(model, args.model_out)
    print(f"Saved model: {output_path}")


if __name__ == "__main__":
    main()
