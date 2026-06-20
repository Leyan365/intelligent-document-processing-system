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
- `evaluation/latency_eval.py` for CPU latency benchmarking with per-stage
  timings and aggregate runtime statistics.
- `evaluation/layout_feature_eval.py` for comparing text-only classification
  against text plus CPU-friendly layout-proxy features.
- Extraction smoke tests through the document-type-aware extractor examples.
- Validation boundary smoke tests through `validation.py` examples covering
  clean, noisy, low-confidence, and invalid-field cases.

## Lecturer Feedback Addressed

The project has addressed the main lecturer feedback items that apply to the
core IDP and evaluation pipeline:

- Validation boundary and confidence matrix were added to reduce cascade error
  propagation.
- Semantic search is now evaluated mathematically with Precision@K, Recall@K,
  MRR@K, and NDCG@K.
- CPU feasibility is addressed through a stage-level latency benchmark
  framework.
- The text-only classifier limitation is addressed through a lightweight
  layout-aware comparison experiment.

## Final Evaluation Rerun Tasks

- Rerun semantic search evaluation in the production SBERT + FAISS environment.
- Rerun the latency benchmark with real invoice, receipt, and purchase order
  files in the full dependency environment.
- Repeat layout-aware comparison on a more diverse same-domain document split if
  additional real samples are available.
- Capture final metrics and screenshots for the dissertation and final
  presentation.
- Optional: run a Tesseract OCR baseline or fallback comparison.

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

This turns semantic search into a measurable retrieval component rather than a
UI-only capability. Future work is to rerun it using the production SBERT +
FAISS environment.

## Phase 13: CPU Latency Benchmark

Benchmark framework implemented:

```text
evaluation/latency_eval.py
```

Implemented benchmark stages:

- file loading and text extraction;
- OCR fallback where applicable;
- document classification;
- field extraction;
- validation;
- embedding generation;
- FAISS indexing;
- end-to-end processing time.

The script reports per-document timings, averages, median/min/max total runtime,
slowest document, slowest stage, CSV output when requested, and skipped
dependency warnings. Future work should repeat file-mode benchmarking with real
PDF/image samples in an environment where PyMuPDF, PaddleOCR,
sentence-transformers, and FAISS are installed.

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

The current production classifier remains unchanged. The Phase 14 evaluation
adds a CPU-friendly research comparison using numeric text-structure features
such as line counts, top/bottom-region keyword counts, amount/date patterns,
payment keywords, and table-density proxies. On the current validation split,
the layout-aware model matched the text-only baseline, suggesting the current
split is already highly separable by text/domain cues. Future work should repeat
the experiment on more diverse same-domain documents where spatial structure may
provide stronger additional signal.

## Optional Tesseract Baseline

An optional OCR comparison can benchmark PaddleOCR against Tesseract for a small
sample of scanned PDFs/images. The comparison should focus on:

- text extraction quality;
- downstream classification impact;
- downstream field extraction impact;
- OCR runtime on CPU.

This is optional because the current pipeline is already implemented with
PaddleOCR and because the core remaining dissertation feedback is centered on
retrieval metrics, latency, layout-aware comparison, and final real-environment
reruns.