# Intelligent Document Processing System

A local-first Intelligent Document Processing (IDP) system for final-year BSc
Data Science project work. The system extracts text from business documents,
classifies document type, extracts key fields, validates pipeline reliability,
indexes processed documents for semantic search, and displays results through a
Streamlit interface.

The core IDP and evaluation pipeline is complete through Phase 14. The remaining
implementation work focuses on application-level persistence, authentication,
and duplicate protection.

## Current Features

Completed pipeline and evaluation features:

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
- Semantic search IR evaluation with Precision@K, Recall@K, MRR@K, and NDCG@K.
- CPU latency benchmark framework for stage-level runtime measurement.
- Layout-aware classification comparison using lightweight text-structure
  features.
- Local Streamlit application for upload, processing, result review, search,
  and processing history.

Remaining must-have application features:

- Authentication/login and user session management.
- Persistent database storage for processed document records.
- Duplicate upload protection using file hashing.

## Pipeline Summary

Implemented prototype pipeline:

```text
Upload document
-> text extraction / OCR
-> document classification
-> field extraction
-> validation and confidence matrix
-> semantic search indexing
-> Streamlit display
```

Planned persistence-aware pipeline:

```text
Authenticated user
-> upload
-> SHA-256 file hash check
-> duplicate detection
-> process new document or reuse existing record
-> database save
-> search indexing/display
```

The top-level orchestrator is `src/idp_system/system.py`. It coordinates the
loader, classifier, extractor, validation layer, semantic search service, and UI
result shape.

## Current Storage Status

Processed documents are currently stored in memory/session state for prototype
review. Persistent storage is planned next.

The recommended database approach is SQLite first for local dissertation/demo
reliability and quick duplicate-protection integration. MySQL can be used if the
existing proposal or supervisor requirements strictly require it. The schema
should stay database-agnostic where possible so it can be moved from SQLite to
MySQL later with minimal changes.

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
  search_eval.py
  latency_eval.py
  layout_feature_eval.py
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

```cmd
set PYTHONPATH=src
python -m streamlit run src\idp_system\ui\streamlit_app.py
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
$env:PYTHONPATH='src'; python evaluation/search_eval.py
$env:PYTHONPATH='src'; python evaluation/latency_eval.py
$env:PYTHONPATH='src'; python evaluation/layout_feature_eval.py
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

- Authentication/login is not implemented yet and is the next required
  application-level feature.
- Persistent database storage is not implemented yet; processed records are
  currently in-memory/session-based for prototype review.
- Duplicate upload protection is not implemented yet and should be handled with
  SHA-256 file hashing.
- Validation is advisory only; it flags low-confidence or suspicious outputs but
  does not block downstream processing.
- The trained classifier reports perfect validation accuracy on the current
  local split, but that result should be treated cautiously because document
  classes come from different source domains.
- Purchase order training data is much smaller than receipt training data.
- RVL-CDIP invoice samples depend on OCR quality and can be noisy.
- Semantic search, latency, and layout-aware evaluation frameworks are
  implemented, but final reruns should be performed in the full production
  dependency environment with real representative files.

## Changes From Original Proposal

- The Flask backend was deferred in favor of a Streamlit-first implementation.
- Authentication and database integration were initially deferred during ML
  pipeline development, but they are now required before final submission.
- MySQL was proposed originally. SQLite may be used for local prototype
  persistence if acceptable, while keeping the schema portable to MySQL.
- BGE-M3 was reduced to Sentence-BERT/MiniLM-style local embeddings for
  feasibility.
- Additional academic evaluation phases were added beyond the proposal:
  validation matrix, MRR/NDCG evaluation, CPU latency benchmark, and
  layout-aware classification comparison.

## Next Planned Phases

- Phase 15: authentication and user session management.
- Phase 16: persistent database storage and duplicate upload protection.
- Phase 17: final real-environment evaluation reruns and screenshots.
- Phase 18: dissertation/report writing and final presentation preparation.