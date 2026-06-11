# Evaluation Plan

## Completed Evaluation Work

The project currently includes lightweight evaluation and smoke-test coverage
for the main pipeline stages:

- Classification validation through the trained three-class classifier and
  validation split output.
- `evaluation/classification_eval.py` for RVL-CDIP invoice detection sanity
  checks when local data is available.
- `evaluation/extraction_eval.py` for approximate extraction checks on FUNSD
  samples when local data is available.
- `evaluation/search_eval.py` for semantic search evaluation with Precision@K,
  Recall@K, MRR@K, and NDCG@K.
- Extraction smoke tests through the document-type-aware extractor examples.
- Validation boundary smoke tests through `validation.py` examples covering
  clean, noisy, low-confidence, and invalid-field cases.

## Lecturer Feedback Addressed

The validation boundary and confidence matrix have been added. The current
system now exposes:

- OCR/text quality warnings;
- classification confidence and confidence source;
- field-level validation;
- total warning counts;
- critical warning counts;
- validation score;
- overall pipeline status.

This directly addresses the need to reduce cascade error propagation by making
pipeline reliability visible to the reviewer.

## Lecturer Feedback Still Planned

- Repeat semantic search evaluation using the production SBERT + FAISS
  environment.
- CPU feasibility benchmark with stage-level latency.
- Layout-aware feature comparison against the current text-only classifier.
- Optional Tesseract OCR baseline or fallback comparison.

## Phase 12: Semantic Search Evaluation

Evaluation framework implemented:

```text
evaluation/search_eval.py
```

Implemented work:

- Built a small relevance dataset with queries and expected relevant document
  IDs.
- Indexed the evaluation documents through the existing semantic search service.
- Ran each query at fixed `k` values.
- Computed Precision@K, Recall@K, MRR@K, and NDCG@K.
- Reported aggregate metrics and weak-performing queries for dissertation
  analysis.

This will turn the existing semantic search feature into a measurable retrieval
component rather than a UI-only capability.

Future work: repeat the evaluation using the production SBERT + FAISS
environment.

## Phase 13: CPU Latency Benchmark

Planned benchmark stages:

- file loading and text extraction;
- OCR fallback where applicable;
- document classification;
- field extraction;
- validation;
- embedding generation;
- FAISS indexing;
- end-to-end processing time.

Expected output should include per-document timings, averages, and slowest-stage
analysis. The goal is to support the dissertation claim that the system is
feasible for local CPU-first processing and to identify the largest latency
bottlenecks.

## Phase 14: Layout-Aware Feature Comparison

Planned comparison:

```text
text-only classifier
vs
text + layout/spatial features
```

The current classifier uses only text features. Phase 14 should test whether
layout-aware features improve classification reliability, especially for noisy
OCR documents where field positions and document structure may carry useful
signals.

## Optional Tesseract Baseline

An optional OCR comparison can benchmark PaddleOCR against Tesseract for a small
sample of scanned PDFs/images. The comparison should focus on:

- text extraction quality;
- downstream classification impact;
- downstream field extraction impact;
- OCR runtime on CPU.

This is optional because the current pipeline is already implemented with
PaddleOCR and because the core remaining dissertation feedback is centered on
retrieval metrics, latency, and layout-aware comparison.
