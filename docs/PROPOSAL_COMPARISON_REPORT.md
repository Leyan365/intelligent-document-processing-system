# Intelligent Document Processing (IDP) System
## Proposal vs. Implementation Comparative Analysis Report

---

### Executive Summary

This report presents a comparative analysis between the original project proposal (**"Intelligent Document Processing System with Machine Learning-Based Information Extraction and Semantic Search"**, `Project Proposal.docx`) and the actual implementation realized in the **`IDP-System`** codebase.

The system was designed as a local-first, AI-driven Intelligent Document Processing pipeline targeting semi-structured and unstructured business documents—specifically **Invoices**, **Receipts**, and **Purchase Orders**.

Every core functional requirement has been implemented, with dedicated evaluation frameworks covering the principal machine-learning, information-retrieval, and performance components. Furthermore, several specialized research and pipeline safeguard components were introduced to enhance academic depth, pipeline reliability, computational transparency, and user review workflows.

---

### 1. High-Level Comparison Matrix

| # | Proposal Component | Proposed Specification | Implementation Status | Implemented Architecture & Technologies |
|---|---|---|---|---|
| **1** | **User Authentication** | User registration and login; access control | **Implemented & Refined** | Local academic prototype authentication: salted PBKDF2-HMAC-SHA256 password hashing (200k iterations), opaque database-backed session tokens (`auth_sessions`), session expiry/revocation, and per-user data scoping. |
| **2** | **Document Input & Preprocessing** | Digital PDF parsing & scanned image/PDF preprocessing with OpenCV (grayscale, noise removal, thresholding, deskewing) | **Implemented & Enhanced** | `loader.py` + `preprocessing.py`: PyMuPDF (`fitz`) direct text extraction with 50-char fallback threshold; page rendering at 300 DPI; OpenCV median blur, adaptive Gaussian thresholding, and `minAreaRect` deskewing; SHA-256 duplicate detection. |
| **3** | **Text Extraction (OCR)** | PaddleOCR for scanned documents with text cleaning and normalization | **Implemented** | `ocr.py`: PaddleOCR wrapper with lazy loading, angle classification (`use_angle_cls=True`), multi-version compatibility fallback, text cleaning. |
| **4** | **Document Classification** | Supervised ML (TF-IDF + Logistic Regression) on Invoices, Receipts, and Purchase Orders; LayoutLMv3 exploration | **Implemented & Enhanced** | `classifier.py` + `training/`: Trained TF-IDF + Logistic Regression classifier with class balancing; hybrid heuristic override rules for high-confidence cues; Phase 14 layout-aware 25-feature proxy comparison experiment. |
| **5** | **Information Extraction** | Hybrid regex + NLP (spaCy) for Invoice Number, Date, Total Amount, Supplier | **Implemented & Enhanced** | `extractor.py`: Document-type-aware extraction rules tailored to Invoices, Receipts, and POs; currency/numeric regex; date parsing across 11 formats; spaCy NER fallback (`ORG`, `DATE`, `MONEY`); noise filtering. |
| **6** | **Advisory Validation Boundary** | *Not explicitly planned in proposal* | **Added (Key Enhancement)** | `validation.py` (Phase 11): Multi-stage advisory validation checking text quality/noise, classification confidence, and field syntax validity to prevent cascading errors. |
| **7** | **Data Storage & Persistence** | Relational Database (MySQL) for metadata, classification, and extracted fields | **Implemented (SQLite, Portable Schema)** | `database/` (Phase 15/16): SQLite persistence (`data/app/idp_app.db`) tracking `users`, `auth_sessions`, `documents`, `classifications`, `extracted_fields`, `validation_results`, and SHA-256 file hashes. Schema is designed to facilitate future MySQL migration. |
| **8** | **Semantic Search** | Embeddings (Sentence-BERT or BGE-M3) + FAISS index; natural language document retrieval | **Implemented & Enhanced** | `embeddings.py` + `search.py`: `sentence-transformers/all-MiniLM-L6-v2` generating 384-d normalized embeddings + FAISS `IndexFlatIP` (cosine similarity); structured document search context blocks; in-memory query engine with offline fallback. |
| **9** | **User Interface** | Web application with Flask backend and Streamlit dashboard | **Implemented (Streamlit-First)** | `ui/streamlit_app.py`: Integrated Streamlit application covering Authentication, Document Ingestion, Live Processing Stages, Interactive Field Editing, Validation Confidence Matrix, Semantic Search, and Document History. |
| **10** | **Evaluation & Benchmarking** | Basic accuracy, F1-score, precision, recall, relevance check | **Frameworks Implemented** | Full evaluation suite (`evaluation/`): Classification F1/Confusion Matrix, Information Extraction accuracy, Mathematical IR metrics (MRR@K, NDCG@K, Precision@K, Recall@K), CPU Stage-by-Stage Latency Profiler, and Layout-Aware comparison. |

---

### 2. Detailed Stage-by-Stage Architectural Breakdown

#### 2.1 Document Ingestion & Preprocessing
* **Proposal Plan**: Accept digital PDFs and scanned images/PDFs. Use PyMuPDF for direct digital PDF parsing and OpenCV for image preprocessing (grayscale conversion, noise filtering, adaptive thresholding, and deskewing).
* **Implementation Details**:
  * Implemented `DocumentLoaderRouter` in `src/idp_system/pipeline/loader.py` supporting `PDF`, `IMAGE` (PNG, JPG, JPEG, TIFF, BMP), `TXT`, `DOCX`, `MD`, `HTML`, `JSON`, and `CSV`.
  * Digital PDFs are first parsed with PyMuPDF (`fitz`). If extracted text is under 50 characters (indicative of a scanned or image-based PDF), pages are dynamically rendered as 300 DPI pixmaps and routed to the OCR engine.
  * `ImagePreprocessor` in `src/idp_system/pipeline/preprocessing.py` implements OpenCV grayscale conversion (`cv2.cvtColor`), median blur denoising (`cv2.medianBlur(ksize=3)`), adaptive Gaussian thresholding (`cv2.adaptiveThreshold`), and minimum-area bounding box deskewing (`cv2.minAreaRect` + `cv2.warpAffine`).
  * **Addition**: Integrated SHA-256 file hashing on upload to detect duplicates immediately per user before CPU-heavy OCR execution.

#### 2.2 Optical Character Recognition (OCR)
* **Proposal Plan**: Extract machine-readable text from scanned documents using PaddleOCR; normalize and clean output text.
* **Implementation Details**:
  * Implemented `OCRService` in `src/idp_system/pipeline/ocr.py`.
  * Features lazy singleton initialization to reduce startup overhead.
  * Configured with angle classification (`use_angle_cls=True`) and robust fallback logic handling multiple PaddleOCR versions and keyword compatibility differences (`show_log`, `use_doc_orientation_classify`, etc.).
  * Applies text cleaning and normalization to remove OCR artifacts, broken line breaks, and excessive spacing.

#### 2.3 Document Classification & Layout-Aware Research
* **Proposal Plan**: Train a supervised ML classifier (TF-IDF + Logistic Regression) on Invoices, Receipts, and Purchase Orders. Consider layout-aware models (LayoutLMv3) as an advanced extension if feasible.
* **Implementation Details**:
  * Implemented `DocumentClassifier` in `src/idp_system/pipeline/classifier.py` and training pipeline in `training/train_document_classifier.py`.
  * Uses `TfidfVectorizer` (unigrams + bigrams, English stop words) and `LogisticRegression` (`max_iter=1000`, `class_weight="balanced"`, deterministic random state `42`).
  * Persisted model artifact saved as `models/document_classifier.joblib`.
  * **Hybrid Architecture**: Combines ML probability confidence scoring (`predict_proba`) with strong domain heuristics (`heuristic_document_type`) for deterministic high-precision routing.
  * **Layout-Aware Extension (Phase 14)**: To explore the LayoutLMv3 proposal objective in a CPU-friendly manner without heavy multimodal transformer dependencies, `evaluation/layout_feature_eval.py` was developed. It evaluates a hybrid model combining TF-IDF with 25 structural layout-proxy features (line counts, density proxies, upper/lower quadrant keyword distributions, currency counts, table density metrics).

#### 2.4 Information Extraction
* **Proposal Plan**: Extract core business fields—**Invoice Number**, **Date**, **Total Amount**, and **Supplier Name**—using a hybrid approach combining regular expressions, pattern matching, and spaCy NLP.
* **Implementation Details**:
  * Implemented `InformationExtractor` in `src/idp_system/pipeline/extractor.py`.
  * **Document-Type-Aware Logic**: Custom extraction routines tailored to the specific structure of `invoice`, `receipt`, and `purchase_order` documents.
  * **Invoice Number / Document ID**: Extracts invoice numbers (`INV-\d+`), PO numbers (`PO\d+`, `PO-?\d+`), and receipt transaction IDs while discarding noise tokens.
  * **Date Extraction**: Matches and normalizes 11 date formats (`YYYY-MM-DD`, `DD/MM/YYYY`, `MM/DD/YYYY`, `DD-Mon-YYYY`, `Month DD, YYYY`, etc.).
  * **Amount Extraction**: Extracts monetary values with currency symbol support (`$`, `Rs.`, `RM`), prioritizing total/balance due fields and ignoring tax/subtotal line items.
  * **Supplier Extraction**: Multi-stage extraction using header analysis, labeled patterns (`Supplier:`, `Vendor:`, `From:`), company header heuristic detection, and spaCy Named Entity Recognition (`ORG` entities), backed by an extensive stop-marker filter (`BAD_SUPPLIER_MARKERS`).

#### 2.5 Advisory Validation Boundary (Pipeline Safeguard)
* **Proposal Plan**: Not explicitly planned in original proposal.
* **Implementation Details (Phase 11 & 11B)**:
  * Implemented in `src/idp_system/pipeline/validation.py` to address cascading error propagation.
  * Computes an advisory validation score ($0.0 - 1.0$) and categorizes documents into:
    * `processed` (all checks passed)
    * `processed_with_warnings` (minor non-critical anomalies)
    * `needs_review` (critical errors detected)
  * Validates:
    * **Text Quality**: Character/word counts, symbol noise ratio, digit ratio, uppercase ratio.
    * **Classification Confidence**: Flags heuristic vs. ML sources, warning on low ML confidence ($< 0.45$).
    * **Field Integrity**: Validates date chronological boundaries ($1990 - 2100$), numeric amount formats, and non-empty supplier/ID values.

#### 2.6 Semantic Search & Information Retrieval
* **Proposal Plan**: Vector embeddings using Sentence-BERT or BGE-M3; similarity search indexed with FAISS; support natural language semantic queries.
* **Implementation Details**:
  * Implemented `EmbeddingService` in `src/idp_system/pipeline/embeddings.py` using `sentence-transformers/all-MiniLM-L6-v2` producing 384-dimensional normalized dense vectors.
  * Implemented `SemanticSearchService` in `src/idp_system/pipeline/search.py` using FAISS `IndexFlatIP` (inner product on normalized vectors $\equiv$ cosine similarity).
  * Constructs structured document representation blocks combining predicted document type, supplier, document ID, date, amount, and text preview for retrieval relevance.
  * Features automatic in-memory index reconstruction on application startup from persisted user records.
  * Offline fallback index (`SimpleInnerProductIndex`) ensures zero-crash operation when compiled FAISS binaries are unavailable.

#### 2.7 Database Persistence & User Management
* **Proposal Plan**: Relational database (MySQL) to store document metadata, classification results, and extracted fields. User registration and authentication.
* **Implementation Details**:
  * Implemented relational persistence in `src/idp_system/database/` and `src/idp_system/auth.py`.
  * **Technology Adjustment**: Implemented using SQLite (`data/app/idp_app.db`) for self-contained, zero-dependency local academic evaluation. The schema and repository design are intentionally database-portable to facilitate future MySQL migration.
  * Implemented Tables:
    * `users`: `user_id`, `username`, `email`, `password_hash`, `salt`, `created_at`, `last_login_at`.
    * `auth_sessions`: `session_id`, `user_id`, `token_hash`, `created_at`, `expires_at`, `revoked_at`.
    * `documents`: `document_id`, `user_id`, `file_hash` (SHA-256), `original_filename`, `stored_path`, `raw_text`, `result_json`, `created_at`.
    * `classifications`: `classification_id`, `document_id`, `label`, `confidence`, `confidence_source`.
    * `extracted_fields`: `field_id`, `document_id`, `field_name`, `field_value`.
    * `validation_results`: `validation_id`, `document_id`, `pipeline_status`, `validation_score`, `warnings_json`.
  * **Duplicate Prevention**: Enforced via `UNIQUE(user_id, file_hash)` constraint, allowing instant retrieval of previously processed documents without redundant computation.

#### 2.8 User Interface & Experience
* **Proposal Plan**: Web application with Flask backend and Streamlit frontend dashboard for upload, data display, and semantic search.
* **Implementation Details**:
  * Implemented in `src/idp_system/ui/streamlit_app.py` as an integrated Streamlit application.
  * **Architecture Rationale**: Consolidated into a Streamlit-first architecture rather than a separate Flask API server to simplify local deployment, reduce service-management complexity, give direct access to local processing state, and simplify integration with the in-memory FAISS index.
  * **UI Modules**:
    1. *Authentication View*: Sign-in and Sign-up tabs with validation and persistent session cookies.
    2. *Document Upload & Processing*: File drag-and-drop, step-by-step progress tracking, OCR preview, and document type confidence indicators.
    3. *Extracted Fields & In-Place Editing*: Structured key-value cards with live field editing capabilities.
    4. *Validation Matrix*: Visual indicators of document quality, noise metrics, and pipeline warnings.
    5. *Semantic Search*: Real-time query input, ranked search results, relevance similarity score cards, and field badge summaries.
    6. *Document History*: Historical list of processed documents per user with inspection, preview, and original file download.

---

### 3. Comprehensive Evaluation Suite (Additions Beyond Proposal)

While the proposal outlined basic accuracy, F1, and recall metrics, the implementation incorporates five dedicated evaluation frameworks under `evaluation/`:

```
evaluation/
├── classification_eval.py     # Multiclass precision, recall, F1, and confusion matrix
├── extraction_eval.py         # Field-level extraction sanity benchmarking
├── search_eval.py             # Information Retrieval evaluation (MRR@K, NDCG@K, Precision@K, Recall@K)
├── latency_eval.py            # Stage-by-stage CPU runtime profiler
├── layout_feature_eval.py     # Text-only vs. Text + Layout Proxy comparison
└── utils.py                  # Evaluation utilities and metric helpers
```

#### 3.1 Information Retrieval Metrics for Semantic Search (`search_eval.py`)
Phase 12 introduced formal mathematical IR evaluation:
* **Precision@K & Recall@K**: Measuring fraction of relevant documents retrieved in top-$K$.
* **Mean Reciprocal Rank (MRR@K)**: Measuring rank position of the first relevant result ($MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$).
* **Normalized Discounted Cumulative Gain (NDCG@K)**: Evaluates graded relevance ($0-3$) with logarithmic rank discounting ($DCG@K = \sum_{i=1}^K \frac{2^{rel_i}-1}{\log_2(i+1)}$).

#### 3.2 Stage-Level CPU Latency Profiler (`latency_eval.py`)
Profiles stage execution times using `time.perf_counter()` to establish CPU feasibility:
* Times file loading & PyMuPDF parsing
* Times OpenCV preprocessing & PaddleOCR inference
* Times TF-IDF vectorization & Logistic Regression classification
* Times Information Extraction rules & spaCy NER
* Times Advisory Validation checks
* Times Sentence-BERT embedding generation & FAISS indexing
* Computes Mean, Median, Min, Max, and identifies computational bottlenecks.

#### 3.3 Layout-Aware vs. Text-Only Classifier Evaluation (`layout_feature_eval.py`)
Addresses the proposal's research question regarding spatial structure understanding:
* Extracts 25 dense structural features from document text geometry.
* Compares standard text-only TF-IDF against TF-IDF + standardized layout features.
* Evaluates macro/weighted precision, recall, F1-scores, and top informative layout features per class.

---

### 4. Technology Stack Comparison

| Category | Proposed in Proposal | Implemented in Codebase | Notes / Justification |
|---|---|---|---|
| **Programming Language** | Python 3.10+ | Python 3.10+ (`dsenv` / Conda) | Fully aligned. |
| **PDF Processing** | PyMuPDF | PyMuPDF (`fitz` 1.23+) | Fully aligned. Direct text extraction + 300 DPI pixmap rendering. |
| **Computer Vision** | OpenCV | `opencv-python` (`cv2` 4.8+) | Fully aligned. Grayscale, median blur, adaptive threshold, deskewing. |
| **OCR Framework** | PaddleOCR | `paddleocr` + `paddlepaddle` | Fully aligned. Version-compatible fallback handling. |
| **Machine Learning** | scikit-learn | `scikit-learn` + `joblib` | Fully aligned. TF-IDF + Logistic Regression pipeline. |
| **NLP & NER** | spaCy | `spacy` + `re` (RegEx) | Fully aligned. Hybrid regex rules with spaCy NER fallback. |
| **Embeddings** | Sentence-BERT / BGE-M3 | `sentence-transformers` (`all-MiniLM-L6-v2`) | Selected MiniLM-L6-v2 as a lightweight and CPU-friendly model for standard computing environments. |
| **Vector Search** | FAISS | `faiss-cpu` | Fully aligned. `IndexFlatIP` with cosine similarity on normalized vectors. |
| **Persistence / DB** | MySQL | SQLite (`idp_app.db`) | SQLite adopted for self-contained local academic evaluation; schema design is portable to facilitate future MySQL migration. |
| **Web Framework** | Flask + Streamlit | Streamlit (Integrated) | Consolidated into Streamlit-first architecture to simplify local deployment and state management. |
| **Security & Auth** | Basic login | Salted PBKDF2-HMAC-SHA256 | Local academic prototype: salted PBKDF2-HMAC-SHA256 password hashing and opaque database-backed session tokens. |
| **Duplicate Check** | *Not specified* | SHA-256 File Hashing | Added to avoid redundant OCR and embedding computation. |
| **Evaluation** | Basic metrics | Precision/Recall/F1, MRR@K, NDCG@K, CPU Latency, Layout Proxy | Significantly expanded academic evaluation frameworks. |

---

### 5. Datasets & Model Training Breakdown

| Dataset | Proposed Role | Implementation Role & Details |
|---|---|---|
| **RVL-CDIP (Invoices)** | Document classification | Cached OCR text extracted via `training/build_rvl_text_cache.py`. |
| **SROIE (Receipts)** | Document understanding / classification | OCR text extracted from ground truth bounding boxes. |
| **Real Purchase Orders** | PO domain classification | Real PO PDF collection processed via `training/build_po_text_cache.py`. |
| **Custom Text Dataset** | Unified 3-class dataset | Consolidated 3-class train/val dataset assembled via `training/build_custom_text_dataset.py`. |
| **FUNSD** | Form understanding exploration | Referenced for semi-structured form validation checks (`evaluation/extraction_eval.py`). |

---

### 6. Architectural Adjustments & Rationale

1. **Streamlit-First vs. Separate Flask Backend**:
   * *Proposed*: Flask API backend communicating with a separate Streamlit frontend.
   * *Implemented*: Fully integrated Streamlit application.
   * *Rationale*: Simplifies local deployment, eliminates multi-process management during local defense, and enables direct in-memory vector index synchronization.
2. **SQLite vs. MySQL**:
   * *Proposed*: MySQL database.
   * *Implemented*: SQLite database (`idp_app.db`) with portable relational schema.
   * *Rationale*: Guarantees zero-configuration reproducibility on examiner machines without requiring a live MySQL daemon running. The schema and query structure are structured to facilitate future MySQL migration.
3. **Embedding Model Selection (all-MiniLM-L6-v2)**:
   * *Proposed*: Sentence-BERT or BGE-M3.
   * *Implemented*: `sentence-transformers/all-MiniLM-L6-v2`.
   * *Rationale*: Provides 384-dimensional dense vectors with lightweight, CPU-friendly embedding inference suitable for standard personal computer hardware.

---

### 7. Important Academic Limitations & Research Evidence Gaps

To maintain honest academic rigor for dissertation reporting, the following known constraints are acknowledged:
1. **Dataset Domain Imbalance & Source Separation**: Invoices, receipts, and purchase orders originate from different source datasets (RVL-CDIP, SROIE, real PO PDFs). The classifier's high reported validation accuracy should be interpreted cautiously as domain artifacts may contribute to separability.
2. **Purchase Order Sample Size**: Purchase order training samples (18 train, 6 val) are much smaller than receipt samples (626 train, 347 val).
3. **In-Memory Search Vectors**: Semantic search embeddings are indexed in memory and rebuilt from persisted SQLite records on startup rather than being stored in a dedicated vector database.
4. **Advisory-Only Validation**: The validation boundary provides warnings and confidence scoring but is advisory and does not hard-block downstream pipeline execution.
5. **Academic Prototype Authentication**: Password hashing and session cookies follow sound cryptographic principles but represent a local academic prototype without multi-factor authentication or role-based access control.
6. **Empirical Rerun Tasks**: While evaluation frameworks (MRR/NDCG, CPU latency profiler, layout-aware classifier) are fully implemented, final empirical rerun sweeps in the complete dependency environment remain for dissertation data collection.

---

### 8. Summary of Deliverables Status

* [x] **Software System**: End-to-end web application with authentication, ingestion, OCR, classification, extraction, validation, semantic search, and history.
* [x] **Document Processing Pipeline**: Multi-stage pipeline coordinating PyMuPDF, OpenCV, PaddleOCR, TF-IDF, spaCy/RegEx, and validation.
* [x] **Machine Learning Models**: Trained three-class document classifier (`document_classifier.joblib`) and normalized embedding service.
* [x] **Relational Persistence**: Functional multi-table relational schema with duplicate detection and user isolation.
* [x] **Semantic Search**: FAISS vector search over structured document representations.
* [x] **Academic Evaluation Frameworks**: Evaluation suite covering Classification F1/Confusion Matrices, Search IR metrics (MRR/NDCG), CPU Latency benchmarks, and Layout-Aware experiments.

---
*Report synchronized for the Intelligent Document Processing (IDP) System repository.*
