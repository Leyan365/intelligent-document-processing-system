# Project Status

## Current Status After Phase 16

The core IDP pipeline is complete. Phase 15 added local Streamlit authentication, and Phase 16 added persistent SQLite document storage with per-user duplicate upload protection. The remaining work focuses on improvement/tuning, UI polish, final real-environment evaluation reruns, screenshots, dissertation/report writing, and presentation preparation.

The current system can process invoices, receipts, and purchase orders through
the local pipeline:

```text
Upload document
-> text extraction / OCR
-> document classification
-> field extraction
-> validation and confidence matrix
-> semantic search indexing
-> Streamlit display
```

The project also includes formal evaluation scripts for semantic search,
latency benchmarking, and layout-aware classification comparison.

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
- Phase 12: Added semantic search evaluation with Precision@K, Recall@K,
  MRR@K, and NDCG@K.
- Phase 13: Added CPU latency benchmark framework.
- Phase 14: Added layout-aware classification comparison.
- Phase 15: Added local SQLite-backed Streamlit authentication and user session
  management.
- Phase 16: Added persistent SQLite document storage and per-user duplicate
  upload protection.

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
  details, search, and persisted per-user history.
- `evaluation/search_eval.py` evaluates semantic search with IR metrics.
- `evaluation/latency_eval.py` benchmarks stage-level CPU latency.
- `evaluation/layout_feature_eval.py` compares text-only classification with
  text plus layout-proxy features.

## Release Tag

- `v1.0-final-project` -> points to commit `379d0d9`.

## Recent Commits

- 96b4263 Phase 11: Add validation boundary and confidence matrix.
- e8f34ef Phase 11B: Improve receipt and purchase order validation
  reliability.
- a57b6fe Phase 12: Add semantic search evaluation metrics.
- 7fc7e00 Phase 13: Add CPU latency benchmark.
- 379d0d9 Phase 14: Add layout-aware classification comparison.
- 4cdb5b3 Phase 15: Add local authentication and fix PDF upload handling.
- 54402ba Phase 16: Add persistent document storage and duplicate protection.

## Known Limitations

- Authentication is implemented as a local SQLite-backed academic prototype. It does not include MFA or RBAC.
- Processed document records persist in SQLite with per-user duplicate protection.
- Semantic search embeddings are still in-memory and are rebuilt from persisted
  documents for the signed-in user when needed.
- Validation is advisory only and does not block classification, extraction, or
  indexing.
- The classifier's reported validation accuracy is high, but it may be inflated
  because invoice, receipt, and purchase order samples come from different
  datasets and source domains.
- Receipt samples far outnumber purchase order samples.
- Purchase order validation is based on a small real-PDF sample.
- RVL-CDIP invoice text depends on OCR cache quality.
- Semantic search and latency evaluation frameworks are implemented, but
  production SBERT/FAISS and real-file latency reruns should be performed in the
  full dependency environment.

## Changes From Original Proposal

- The Flask backend was deferred in favor of a Streamlit-first implementation
  so the ML pipeline, validation, and evaluation work could be completed and
  demonstrated quickly.
- Authentication and database integration were initially deferred during ML
  pipeline development, but they are now required before final submission.
- MySQL was proposed originally. SQLite may be used for local prototype
  persistence if acceptable, while keeping the schema portable to MySQL.
- BGE-M3 was reduced to Sentence-BERT/MiniLM-style local embeddings for
  feasibility.
- Additional academic evaluation phases were added beyond the original
  proposal: validation matrix, MRR/NDCG evaluation, CPU latency benchmark, and
  layout-aware classification comparison.

## Next Phases

- Phase 17: Final real-environment evaluation reruns and screenshots.
- Phase 18: Dissertation/report writing and final presentation preparation.
