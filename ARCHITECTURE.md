# Architecture

## System Overview

The project is a local-first Intelligent Document Processing pipeline for
business documents. It avoids external LLM APIs and keeps extraction,
classification, validation, semantic indexing, and display within the local
Python application.

```text
Document upload
-> local text extraction or OCR
-> document classification
-> document-type-aware field extraction
-> advisory validation boundary
-> semantic search indexing
-> Streamlit display
```

The integrated pipeline is coordinated by `IDPSystem` in
`src/idp_system/system.py`.

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
- `streamlit_app.py`: Provides the upload, result review, validation display,
  semantic search, and history UI.

## Data Flow

1. A user uploads a document through Streamlit.
2. The file is saved temporarily and passed to `IDPSystem.process_document`.
3. `DocumentLoaderRouter` selects a local loader based on file extension.
4. PDFs are extracted with PyMuPDF first; low-text PDFs and images use
   PaddleOCR.
5. `DocumentClassifier` predicts one of `invoice`, `receipt`, or
   `purchase_order`.
6. `InformationExtractor` applies document-type-aware extraction rules.
7. `validate_pipeline` checks text quality, classification confidence, and
   extracted field reliability.
8. The processed document is added to the FAISS-backed semantic search service.
9. Streamlit displays extracted fields, validation results, search, and history.

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
- Text-first extraction: Current extraction is based on OCR/text content and
  rules. Layout-aware or spatial features are planned for future comparison.
- In-memory search: Semantic search is useful for prototype review but is not
  yet a persistent production search backend.
