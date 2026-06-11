# Project Status

## Current Status After Phase 11B

The Intelligent Document Processing System is complete through Phase 11B. The
current system can process invoices, receipts, and purchase orders through the
local pipeline:

```text
Upload document
-> text extraction / OCR
-> document classification
-> field extraction
-> validation and confidence matrix
-> semantic search indexing
-> Streamlit display
```

The most recent work added and refined the validation boundary so the UI can
show OCR quality, classification confidence, field-level validation, warnings,
and an overall pipeline status.

## Completed Phases

- Phase 1: Established the project structure and active `src/idp_system/`
  package layout.
- Phase 2: Added document text extraction for local files.
- Phase 3: Added the baseline document classification component.
- Phase 4: Added information extraction for common business-document fields.
- Phase 5: Added semantic search with sentence-transformers and FAISS.
- Phase 6: Integrated the pipeline through the `IDPSystem` orchestrator.
- Phase 7/8: Added the Streamlit UI and classification/extraction heuristic
  polish.
- Phase 9: Added lightweight evaluation scripts for classification and
  extraction baselines.
- Phase 10A: Added RVL-CDIP OCR cache builder.
- Phase 10B: Added custom text dataset builders.
- Phase 10C: Added trained real three-class classifier workflow.
- Phase 10D/10E/10F: Added document-type-aware extraction and reliability
  cleanup.
- Phase 11: Added validation boundary and confidence matrix.
- Phase 11B: Improved receipt and purchase order validation reliability.

## Latest Working Behavior

- Invoice classification and document-type-aware extraction are working.
- Receipt classification and document-type-aware extraction are working.
- Purchase order classification and document-type-aware extraction are working.
- Heuristic overrides improve reliability for strong invoice, receipt, and
  purchase order signals.
- Validation confidence matrix is working and includes text quality,
  classification confidence/source, field validation, warning counts, and
  overall pipeline status.
- Semantic search indexing runs after each processed document and supports
  in-memory query over processed documents.
- Streamlit displays uploaded document results, extracted fields, validation
  details, search, and history.

## Recent Commits

- `96b4263` Phase 11: Add validation boundary and confidence matrix.
- `e8f34ef` Phase 11B: Improve receipt and purchase order validation
  reliability.

## Known Limitations

- Validation is advisory only and does not block classification, extraction, or
  indexing.
- The classifier's reported validation accuracy is high, but it may be inflated
  because invoice, receipt, and purchase order samples come from different
  datasets and source domains.
- Receipt samples far outnumber purchase order samples.
- Purchase order validation is based on a small real-PDF sample.
- RVL-CDIP invoice text depends on OCR cache quality.
- Semantic search has not yet been evaluated with formal relevance metrics.
- CPU latency has not yet been measured as a formal benchmark.
- The current classifier is text-only; layout-aware features are still planned.

## Next Phases

- Phase 12: Semantic search evaluation with Precision@K, Recall@K, MRR@K, and
  NDCG@K.
- Phase 13: CPU latency benchmark for document loading, OCR, classification,
  extraction, validation, embeddings, and indexing.
- Phase 14: Layout-aware feature comparison against the current text-only
  classifier.
- Optional: Tesseract OCR baseline or fallback comparison.
