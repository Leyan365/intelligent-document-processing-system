# Model Training

## Classifier Approach

The current document classifier is a local scikit-learn pipeline:

```text
TF-IDF vectorizer
-> Logistic Regression classifier
```

It predicts three document classes:

- `invoice`
- `purchase_order`
- `receipt`

The implementation uses class balancing and a fixed random seed for repeatable
training behavior.

## Training Script

Training entry point:

```text
training/train_document_classifier.py
```

Default training input:

```text
data/custom_text_dataset/train/
```

Default validation input:

```text
data/custom_text_dataset/val/
```

Default model output:

```text
models/document_classifier.joblib
```

The `models/` directory and `*.joblib` files are ignored by git, so trained
model artifacts stay local.

## Training Command

```powershell
$env:PYTHONPATH='src'; python training/train_document_classifier.py
```

Optional script arguments include:

- `--train-dir`
- `--val-dir`
- `--model-out`
- `--max-per-class`

## Latest Reported Training Output

Training counts:

```text
invoice: 160
purchase_order: 18
receipt: 626
```

Validation counts:

```text
invoice: 40
purchase_order: 6
receipt: 347
```

Accuracy:

```text
1.0000
```

Confusion matrix, with labels ordered as `invoice`, `purchase_order`,
`receipt`:

```text
[[40, 0, 0],
 [0, 6, 0],
 [0, 0, 347]]
```

## Interpretation

The latest reported validation accuracy is 100%, but this should be treated
cautiously. The validation split is useful as a controlled local check, but the
classes are drawn from different source domains:

- RVL-CDIP invoice OCR text for invoices;
- real purchase order PDFs for purchase orders;
- SROIE receipt OCR text for receipts.

Because of this dataset/domain separation, the classifier may be learning some
source-specific patterns in addition to true document-type patterns. A more
rigorous evaluation would include more balanced real-world samples from the same
operational distribution.

## Heuristic Overrides

Heuristic overrides remain important even with the trained classifier. Strong
business-document signals such as `purchase order`, `PO-`, `receipt`, `cashier`,
`invoice number`, and `balance due` can provide reliable decisions and help
protect the pipeline when the ML confidence is unavailable or less informative.

The current classifier therefore uses a hybrid strategy:

```text
strong heuristic signal
-> heuristic label

otherwise
-> TF-IDF + Logistic Regression prediction
```

This is suitable for the current local prototype and aligns with the practical
goal of robust document routing for invoices, receipts, and purchase orders.
