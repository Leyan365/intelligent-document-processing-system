# Architecture

## System Overview

The project is a local-first Intelligent Document Processing pipeline for
business documents. It avoids external LLM APIs and keeps extraction,
classification, validation, semantic indexing, evaluation, and display within
the local Python application.

Current implemented pipeline:

```text
Document upload
-> local text extraction or OCR
-> document classification
-> document-type-aware field extraction
-> advisory validation boundary
-> semantic search indexing
-> Streamlit display
```

Implemented persistence-aware pipeline:

```text
Authenticated user
-> upload
-> SHA-256 file hash check
-> duplicate detection
-> processing or reuse existing record
-> database save
-> search indexing/display
```

The integrated core pipeline is coordinated by `IDPSystem` in
`src/idp_system/system.py`. Authentication, persistent SQLite storage, and
per-user duplicate upload protection are implemented as application-level
layers around the current core pipeline.

## Major Components

- `loader.py`: Routes local input files by extension and extracts text. Digital
  PDFs are processed with PyMuPDF first. If direct PDF text is too short, pages
  are rendered and passed to OCR. Plain text-like files are read directly.
- `preprocessing.py`: Applies OpenCV image preprocessing for OCR, including
  grayscale conversion, denoising, adaptive thresholding, and deskewing.
- `ocr.py`: Wraps PaddleOCR with lazy initialization and compatibility handling
  across PaddleOCR result formats.
- `classifier.py`: Provides TF-IDF plus Logistic Regression document
  classification, trained model loading, confidence reporting, and heuristic
  overrides for invoice, receipt, and purchase order signals.
- `extractor.py`: Extracts common business-document fields with
  document-type-aware rules for invoices, receipts, and purchase orders. Target
  fields are document number, date, amount, and supplier.
- `validation.py`: Runs advisory checks for OCR/text quality, classification
  confidence, and extracted fields. It returns warnings, critical warnings,
  component status, pipeline status, and validation score.
- `embeddings.py`: Generates normalized sentence-transformers embeddings using
  `sentence-transformers/all-MiniLM-L6-v2` by default.
- `search.py`: Stores document embeddings in an in-memory FAISS index and
  returns top-k semantic search results.
- `system.py`: Orchestrates loading, classification, field extraction,
  validation, processed-document storage, and semantic indexing.
- `auth.py`: Provides user authentication, registration, salted PBKDF2 password hashing, and session management.
- `database/`: Implements SQLite repositories for users, authentication sessions, and persisted document records.
- `streamlit_app.py`: Provides the upload, result review, validation display,
  semantic search, and history UI.
- `evaluation/`: Contains evaluation scripts for classification/extraction
  sanity checks, semantic search MRR/NDCG, CPU latency, and layout-aware
  classification comparison.

## Current Data Flow

1. An authenticated user uploads a document through Streamlit.
2. The app reads the uploaded bytes and computes a SHA-256 file hash.
3. The app checks SQLite for an existing document with the same `user_id` and
   `file_hash`.
4. If a duplicate exists, the persisted result snapshot is loaded instead of
   reprocessing.
5. If it is new, the file is saved under `data/app/uploads/<user_id>/` and
   passed to `IDPSystem.process_document`.
6. `DocumentLoaderRouter` selects a local loader based on file extension.
7. PDFs are extracted with PyMuPDF first; low-text PDFs and images use
   PaddleOCR.
8. `DocumentClassifier` predicts one of `invoice`, `receipt`, or
   `purchase_order`.
9. `InformationExtractor` applies document-type-aware extraction rules.
10. `validate_pipeline` checks text quality, classification confidence, and
    extracted field reliability.
11. The processed result, classification, fields, validation metadata, raw text,
    and JSON snapshot are saved to SQLite.
12. The processed document is added to the FAISS-backed in-memory semantic
    search service.
13. Streamlit displays extracted fields, validation results, search, and
    per-user persisted history.

Processed-document records persist in SQLite. Semantic search embeddings remain
in memory and are rebuilt from persisted documents for the signed-in user when
needed.

## Authentication And Persistence Layer

The application implements a local academic prototype authentication and
persistent SQLite document storage layer without changing core IDP model behavior.
User identity is secured using salted PBKDF2-HMAC-SHA256 password hashing (200,000
iterations), and persistent login is managed via opaque browser session tokens
whose SHA-256 digests are tracked in the `auth_sessions` table.

The SQLite schema is kept relational and portable so it can be migrated to MySQL
later if required by external deployment requirements.

Implemented database tables:

- `users`: login identity and password/session metadata.
- `auth_sessions`: persistent session tokens (SHA-256 digest), expiry timestamps, and revocation status.
- `documents`: uploaded document metadata, owner, source filename, file hash,
  processing status, timestamps, and storage references.
- `classifications`: predicted document type, confidence, confidence source,
  and classifier metadata.
- `extracted_fields`: extracted document number, date, amount, supplier, and
  optional normalized field values.
- `validation_results`: pipeline status, validation score, warning counts, and
  serialized warning details.
- Search embeddings are not stored permanently in SQLite; they are rebuilt in
  memory from the current user's persisted document records when needed.

## Duplicate Upload Protection

Duplicate protection is implemented before processing a newly uploaded file:

1. Read uploaded file bytes.
2. Compute a SHA-256 hash.
3. Store `file_hash` in the `documents` table with a `UNIQUE(user_id, file_hash)` constraint.
4. If a user uploads the same file again, return the existing processed record
   instead of reprocessing the document.
5. Duplicate detection is scoped per user, so another user does not see or reuse
   the first user's document history.

This protects CPU-heavy OCR/embedding work from unnecessary repetition and gives
stable document identity for prototype demonstrations.

## Validation Boundary

The validation boundary exists to reduce cascade error propagation. OCR errors,
misclassification, and malformed extracted fields can affect every later stage.
The validation layer makes these risks visible by scoring and warning on:

- extremely short or noisy extracted text;
- unavailable, moderate, or low classification confidence;
- missing or suspicious field values;
- invalid date, amount, supplier, or document number patterns.

This helps a reviewer understand whether downstream outputs should be trusted or
manually checked.

## Advisory Validation Design

Validation is advisory only in the current implementation. It does not block
classification, extraction, indexing, or display. This was chosen because the
system is still an academic prototype and because hard failures could hide
useful partial outputs from the Streamlit reviewer.

Current pipeline statuses are:

- `processed`: no validation warnings.
- `processed_with_warnings`: non-critical warnings are present.
- `needs_review`: at least one validation component failed.

## Design Tradeoffs

- Local-first: The pipeline runs locally without external LLM APIs or cloud OCR
  services.
- CPU-friendly: Core components use PyMuPDF, PaddleOCR, scikit-learn,
  sentence-transformers, and FAISS CPU-compatible workflows.
- Hybrid classification: Strong heuristic overrides are kept alongside the ML
  classifier because business documents often contain reliable keywords such as
  invoice numbers, receipt labels, and purchase order numbers.
- Evaluation-first academic scope: MRR/NDCG, CPU latency, and layout-aware
  comparison have been added as research evidence without rewriting production
  components.
- Database-agnostic persistence: SQLite is used for local reliability, while
  the schema remains portable to MySQL.
- In-memory search: Semantic search vectors are rebuilt from persisted document
  records for the signed-in user instead of being stored permanently.
