# Intelligent Document Processing System

A local-first Intelligent Document Processing (IDP) system for final-year BSc
Data Science project work. The system extracts text from business documents,
classifies document type, extracts key fields, validates pipeline reliability,
indexes processed documents for semantic search, and displays results through a
Streamlit interface.

The core IDP pipeline, authentication, persistence, semantic search, evaluation framework, and polished review UI are implemented. The remaining work focuses on final real-environment evaluation reruns, screenshots, dissertation/report writing, and presentation preparation.

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
- Hybrid semantic search combining deterministic structured-query filtering (amounts, dates, suppliers) with `sentence-transformers/all-MiniLM-L6-v2` embeddings and FAISS.
- Semantic search IR evaluation with Precision@K, Recall@K, MRR@K, and NDCG@K.
- CPU latency benchmark framework for stage-level runtime measurement.
- Layout-aware classification comparison using lightweight text-structure
  features.
- Local Streamlit application for upload, processing, result review, search,
  and processing history.

Completed application and persistence features:

- Local user registration, login, and salted PBKDF2-HMAC-SHA256 password hashing.
- Database-backed opaque authentication sessions and persistent browser cookies.
- Persistent SQLite storage for processed document records and validation results.
- Per-user duplicate upload protection using SHA-256 file hashing.
- Document preview, original-file download, and persistent document review workflow.
- Per-user document history and isolated search index reconstruction.

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

Persistence-aware application pipeline:

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

Processed document records are stored in the local SQLite app database at
`data/app/idp_app.db`. Uploaded files are saved under `data/app/uploads/<user_id>/`, and duplicate uploads are detected per user with SHA-256 file hashes.

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
  database/             SQLite auth and document persistence repositories
  auth.py               user authentication and session management
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
python -m pip install -r requirements.txt
```

`requirements.txt` contains the curated project versions. Scikit-learn is pinned
to 1.7.1 because the current classifier artifact was serialized with that
version. PaddleOCR/PaddlePaddle installation can vary by operating system and
CPU/GPU support; use the CPU build for the local dissertation setup unless GPU
acceleration is explicitly configured.

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

### Persistent Login Security

Persistent login uses a random opaque browser token backed by the local SQLite
`auth_sessions` table. Only a SHA-256 digest of the token is stored in SQLite;
the cookie contains no username, email, password, password hash, or user ID.
Sessions expire after 12 hours by default or after approximately 7 days when
`Keep me signed in` is selected. Sign out revokes the database session and
removes the cookie. Cookie options use `SameSite=Lax` and enable `Secure`
automatically when the app is served over HTTPS.

`streamlit-cookies-controller` is used only because Streamlit 1.55 exposes
cookies through the read-only `st.context.cookies` API. The component sets
cookies in the browser and cannot set `HttpOnly`. This is a documented local
academic-prototype limitation; the cookie therefore remains a strong,
identity-free opaque token and is never placed in URL query parameters.

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

`search_eval.py` defaults to the production
`sentence-transformers/all-MiniLM-L6-v2` plus FAISS path and reports semantic,
entity, identifier, amount, date, mixed hybrid, and no-match categories. Use
`--embedding-backend deterministic` only for offline evaluator testing; those
metrics are labelled fallback/test-only. The current 19-query corpus includes
six semantic-only queries and remains a controlled regression benchmark that
should be expanded before final claims.

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

- Authentication is implemented as a local SQLite-backed academic prototype.
- Semantic search embeddings are still in-memory and rebuilt from persisted
  documents after app restart.
- Hybrid search handles strict numeric inequalities reliably, but pure semantic queries without numeric operators rely on a conservative zero-score threshold.
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

- The separate Flask backend was consolidated into a Streamlit-first integrated
  application to simplify local execution and provide direct state management
  for in-memory FAISS indexing.
- Authentication and database persistence were initially deferred during core ML
  pipeline development, and were subsequently implemented in Phase 15 (auth)
  and Phase 16 (SQLite persistence and duplicate protection), followed by
  Phase 17B session and UI polish.
- MySQL was proposed originally; SQLite is used for local academic evaluation
  and self-contained reproducibility, while maintaining a portable relational
  schema structure.
- BGE-M3 was replaced with `sentence-transformers/all-MiniLM-L6-v2` to provide
  lightweight, CPU-friendly embedding inference on standard personal computers.
- Additional academic evaluation frameworks were introduced beyond the original
  proposal: advisory validation boundary, MRR/NDCG retrieval evaluation,
  stage-level CPU latency benchmarking, and layout-aware classification comparison.

## Remaining Development / Evaluation Work

- Development tuning and quality improvements where academically justified.
- Final production-environment semantic-search evaluation reruns with real embeddings.
- Expand the controlled search benchmark beyond its current 19 queries for
  stronger final dissertation evidence.
- Final real-document latency benchmark measurements in the full dependency environment.
- Dataset and generalization improvements if feasible.
- Final screenshots and reproducibility checks.
- Dissertation/report writing and final presentation preparation.
