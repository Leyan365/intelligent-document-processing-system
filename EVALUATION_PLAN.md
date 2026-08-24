# Evaluation Plan

## Completed Evaluation Work

The project includes formal evaluation and empirical benchmark coverage across the entire IDP pipeline:

- **Classification Validation**: Trained three-class classifier validation split metrics (~100% on development split).
- **Independent Classifier Challenge**: 42-document frozen challenge set (`final_classifier_challenge_v2.csv`) evaluated and documented in [FINAL_CLASSIFIER_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_CLASSIFIER_RESULTS.md) (committed at `43c0a8c`).
- **Information Extraction Benchmark**: 63-document frozen V2 benchmark (`final_extraction_benchmark_v2.csv`) evaluated and documented in [FINAL_EXTRACTION_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_EXTRACTION_RESULTS.md) (committed at `bb9b4a4`).
- **Semantic Search & IR Retrieval**: 19-query hybrid and semantic retrieval evaluation with SBERT MiniLM + FAISS reporting Precision@K, Recall@K, MRR@K, NDCG@K, and negative filter rejection in [FINAL_SEARCH_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_SEARCH_RESULTS.md) (committed at `ca7eea4`).
- **CPU Latency Profiler**: 29-document stage-by-stage runtime benchmark reporting mean/median/min/max execution times, cold-start vs. steady-state costs, and bottleneck analysis in [FINAL_LATENCY_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_LATENCY_RESULTS.md) (committed at `29c7f7f`).
- **Layout-Aware Comparison**: `evaluation/layout_feature_eval.py` comparing text-only classification against text plus 25 CPU-friendly layout-proxy features.
- **Validation Boundary & Smoke Tests**: Multi-rule validation scoring and warning verification across clean, noisy, low-confidence, and invalid-field scenarios.

## Lecturer Feedback Addressed

The project has addressed the main lecturer feedback items that apply to the core IDP and evaluation pipeline:

- Validation boundary and confidence matrix were added to reduce cascade error propagation.
- Semantic search is now evaluated mathematically with Precision@K, Recall@K, MRR@K, and NDCG@K.
- CPU feasibility is addressed through a stage-level latency benchmark framework.
- The text-only classifier limitation is addressed through a lightweight layout-aware comparison experiment.
- Out-of-distribution generalization and cross-domain robustness are explicitly quantified via independent challenge and benchmark manifests.

## Final Evaluation Status (All Complete)

All targeted final empirical evaluations have been executed, verified, and frozen:

1. **Semantic Search Evaluation (Complete)**: Evaluated in the production SBERT (`all-MiniLM-L6-v2`) + FAISS environment across 19 queries (see [FINAL_SEARCH_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_SEARCH_RESULTS.md)).
2. **CPU Latency Benchmark (Complete)**: Evaluated with real invoice, receipt, and purchase order files in the full local dependency environment (see [FINAL_LATENCY_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_LATENCY_RESULTS.md)).
3. **Information Extraction Benchmark (Complete)**: Evaluated against the frozen 63-document V2 manifest (see [FINAL_EXTRACTION_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_EXTRACTION_RESULTS.md)).
4. **Classifier Challenge Evaluation (Complete)**: Evaluated against the frozen 42-document unseen challenge manifest (see [FINAL_CLASSIFIER_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_CLASSIFIER_RESULTS.md)).

## Phase 12: Semantic Search Evaluation

Evaluation framework implemented:

```text
evaluation/search_eval.py
```

Implemented work:

- Built a structured relevance dataset with queries, graded relevance scores, and expected document IDs.
- Indexed the evaluation documents through the semantic search service.
- Computed Precision@K, Recall@K, MRR@K, and NDCG@K across cutoffs $K \in \{1, 3, 5\}$.
- Reported aggregate metrics, category breakdowns, and weak-performing queries for dissertation analysis.
- The production rerun using `sentence-transformers/all-MiniLM-L6-v2` and FAISS is complete and documented in [FINAL_SEARCH_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_SEARCH_RESULTS.md).

## Phase 13: CPU Latency Benchmark

Benchmark framework implemented:

```text
evaluation/latency_eval.py
evaluation/final_latency_eval.py
```

Implemented benchmark stages:

- file loading and PyMuPDF text extraction;
- OpenCV image preprocessing and PaddleOCR inference;
- document classification;
- field extraction;
- validation scoring;
- embedding generation (`all-MiniLM-L6-v2`);
- FAISS index update;
- end-to-end processing time.

The production benchmark across 29 real documents is complete and documented in [FINAL_LATENCY_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_LATENCY_RESULTS.md).

## Phase 14: Layout-Aware Feature Comparison

Comparison implemented:

```text
evaluation/layout_feature_eval.py
```

Implemented comparison:

```text
text-only TF-IDF + Logistic Regression
vs
text TF-IDF + lightweight layout-proxy features + Logistic Regression
```

The current production classifier remains unchanged. The Phase 14 evaluation adds a CPU-friendly research comparison using numeric text-structure features such as line counts, top/bottom-region keyword counts, amount/date patterns, payment keywords, and table-density proxies. On the current validation split, the layout-aware model matched the text-only baseline, indicating that the development split is already highly separable by text/domain cues.

## Optional Tesseract Baseline

An optional OCR comparison can benchmark PaddleOCR against Tesseract for a small sample of scanned PDFs/images. The comparison would focus on text extraction quality, downstream classification/extraction impact, and CPU runtime. This remains an optional extension since PaddleOCR is the primary production engine and all four primary experimental pillars have completed empirical evaluation.

## Phase 18: Hybrid Search Reliability Update (17B.4)

The search pipeline has been augmented with a deterministic structured query parser to correctly handle numeric inequalities, ranges, exact identifiers, and document type constraints that pure embedding similarity fails to enforce reliably.

The retrieval evaluation framework distinguishes between:
- semantic-only queries (natural language semantic intent)
- structured/numeric queries (numeric amount constraint, document type, exact identifier, supplier/entity)
- mixed structured + semantic queries (mixed constraint + semantic intent)

## Phase 19A: Corrected Search Evaluation Harness

The evaluator mirrors the application search schema: `id`, `text`, `type`, `filename`, and nested `fields` for document number, supplier, amount, and date. Malformed records fail before indexing. Reports identify the embedding and index backends, fallback use, retrieval coverage, per-query results, category metrics, overall metrics, and no-match correctness.

The frozen 19-query benchmark was evaluated using `sentence-transformers/all-MiniLM-L6-v2` with FAISS, and full empirical findings are recorded in [FINAL_SEARCH_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_SEARCH_RESULTS.md).

## Frozen Final Extraction Benchmark (V1)

The historical extraction manifest is `evaluation/final_extraction_benchmark.csv` with SHA-256 `04016ac551e0a5dc8a9136085f6025ce7f880113b2953dbc2154d6fa6301b592`. It contains 25 `FINAL_UNSEEN` invoices, 25 `EXTRACTION_ONLY_CLASSIFIER_VAL` SROIE receipts, and 9 `FINAL_UNSEEN` independent purchase orders. The annotation protocol, field normalization, status policy, and planned exact-match evaluation definitions are frozen in `evaluation/FINAL_EXTRACTION_BENCHMARK.md`.

V1 remains immutable historical evidence and was superseded by V2 for final scoring.

## Frozen Final Extraction Benchmark (V2 - Complete)

V2 supersedes V1 for final extraction scoring. The V2 manifest is `evaluation/final_extraction_benchmark_v2.csv`, with SHA-256 `ae7e45d4b273002da476c54208c346cc31f5c53b473adc3900da89bf6a9774d9`. It retains all 59 V1 rows unchanged and adds four `FINAL_UNSEEN` native-text invoice shells that genuinely omit both supplier and document number, adding four `NOT_PRESENT` cases for each field.

V2 contains 63 documents: 29 invoices, 25 receipts, and 9 purchase orders. Full empirical evaluation on V2 was executed in commit `bb9b4a4` and is documented in [FINAL_EXTRACTION_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_EXTRACTION_RESULTS.md).

## Frozen Final Three-Class Classifier Challenge Set (V2 - Complete)

The independent classifier challenge manifest is `evaluation/final_classifier_challenge_v2.csv` with SHA-256 `9c1d629d9a9c32c85c50da7bdc68e81503061ca8d5caf5b886294959f02a90cc`. It contains 42 `FINAL_UNSEEN` documents (18 invoices, 9 purchase orders, and 15 out-of-domain receipts from NOVA IMS). Full empirical evaluation of production pipeline routing and raw ML model accuracy was executed in commit `43c0a8c` and is documented in [FINAL_CLASSIFIER_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_CLASSIFIER_RESULTS.md).
