# Intelligent Document Processing System

A local-first Intelligent Document Processing (IDP) system for final-year BSc
Data Science project work. The system extracts text from business documents,
classifies document type, extracts key fields, validates pipeline reliability,
indexes processed documents for semantic search, and displays results through a
Streamlit interface.

The current implementation is complete through Phase 11B, including invoice,
receipt, and purchase order classification/extraction with advisory validation
and a confidence matrix.

## Current Features

- Digital PDF text extraction with PyMuPDF.
- OCR fallback for scanned PDFs and images with PaddleOCR.
- OCR image preprocessing with OpenCV.
- Three-class document classification for `invoice`, `receipt`, and
  `purchase_order`.
- TF-IDF plus Logistic Regression trained classifier.
- Heuristic overrides for strong invoice, receipt, and purchase order signals.
- Document-type-aware field extraction for document number, date, amount, and
  supplier.
- Advisory validation boundary for OCR quality, classification confidence, and
  extracted fields.
- Validation confidence matrix and warning summary in the Streamlit UI.
- Semantic search with sentence-transformers embeddings and FAISS.
- Local Streamlit application for upload, processing, result review, search,
  and processing history.

## Pipeline Summary

```text
Upload document
-> text extraction / OCR
-> document classification
-> field extraction
-> validation and confidence matrix
-> semantic search indexing
-> Streamlit display
```

The top-level orchestrator is `src/idp_system/system.py`. It coordinates the
loader, classifier, extractor, validation layer, semantic search service, and UI
result shape.

## Project Structure

```text
src/idp_system/
  core/                 configuration, exceptions, logging, document models
  pipeline/             extraction, OCR, classification, validation, search
  ui/                   Streamlit application
  database/             placeholder database adapter
  system.py             integrated IDPSystem orchestrator

training/
  build_rvl_text_cache.py
  build_po_text_cache.py
  build_custom_text_dataset.py
  train_document_classifier.py

evaluation/
  classification_eval.py
  extraction_eval.py
  utils.py
```

Legacy migration code remains under `src/training_data_bot/`, but the active IDP
pipeline is under `src/idp_system/`.

## Setup

Create and activate a virtual environment, then install the project dependencies
needed for the current local pipeline.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pymupdf paddleocr paddlepaddle opencv-python numpy scikit-learn joblib sentence-transformers faiss-cpu streamlit spacy
```

There is currently no committed `requirements.txt`, so dependencies are listed
directly here. PaddleOCR/PaddlePaddle installation can vary by operating system
and CPU/GPU support; use the CPU build for the local dissertation setup unless
GPU acceleration is explicitly configured.

Optional spaCy entity extraction is used only when a local English model is
available. The regex and document-type-aware extraction logic still run without
spaCy.

## Run The Streamlit UI

```powershell
$env:PYTHONPATH='src'; streamlit run src/idp_system/ui/streamlit_app.py
```

The UI supports document upload, processing stage display, extracted field
review, validation results, semantic search over processed documents, and
history viewing.

## Useful Local Checks

Run a basic import check:

```powershell
$env:PYTHONPATH='src'; python -c "from idp_system.system import IDPSystem; print(IDPSystem)"
```

Extract text from one sample file:

```powershell
$env:PYTHONPATH='src'; python scripts/extract_text_sample.py path\to\sample.pdf
```

Run lightweight evaluation scripts when the expected local datasets are present:

```powershell
$env:PYTHONPATH='src'; python evaluation/classification_eval.py
$env:PYTHONPATH='src'; python evaluation/extraction_eval.py
```

## Dataset Building

Build the RVL-CDIP invoice OCR text cache:

```powershell
$env:PYTHONPATH='src'; python training/build_rvl_text_cache.py
```

Build the purchase order text cache:

```powershell
$env:PYTHONPATH='src'; python training/build_po_text_cache.py
```

Build the unified custom text dataset:

```powershell
$env:PYTHONPATH='src'; python training/build_custom_text_dataset.py
```

Expected generated dataset output:

```text
data/custom_text_dataset/
  train/
    invoice/
    purchase_order/
    receipt/
  val/
    invoice/
    purchase_order/
    receipt/
```

## Train The Classifier

Train the local three-class classifier:

```powershell
$env:PYTHONPATH='src'; python training/train_document_classifier.py
```

The script trains a TF-IDF plus Logistic Regression pipeline from
`data/custom_text_dataset/` and writes:

```text
models/document_classifier.joblib
```

The model file is intentionally ignored by git.

## Ignored Local Artifacts

The following are local-only and are not committed:

- `data/`
- `models/`
- `*.joblib`
- `temp_uploads/`
- `sample.pdf`

This keeps large datasets, generated OCR caches, trained model artifacts, and
temporary uploads outside version control.

## Current Limitations

- Validation is advisory only; it flags low-confidence or suspicious outputs but
  does not block downstream processing.
- The trained classifier reports perfect validation accuracy on the current
  local split, but that result should be treated cautiously because document
  classes come from different source domains.
- Purchase order training data is much smaller than receipt training data.
- RVL-CDIP invoice samples depend on OCR quality and can be noisy.
- Semantic search is implemented but still needs formal relevance metrics such
  as MRR and NDCG.
- Latency has not yet been benchmarked formally for CPU-only deployment.
- Layout-aware features have not yet been compared against the current
  text-only classifier.

## Next Planned Phases

- Phase 12: semantic search evaluation with Precision@K, Recall@K, MRR@K, and
  NDCG@K.
- Phase 13: CPU latency benchmark across extraction, OCR, classification,
  field extraction, validation, embedding, and search indexing.
- Phase 14: layout-aware feature comparison against the current text-only
  classifier.
- Optional: Tesseract OCR baseline or fallback comparison.
