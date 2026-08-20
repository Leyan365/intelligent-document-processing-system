# Project Status

## Current Status After Phase 17B

The core IDP pipeline and user interface are complete. Phase 15 added local Streamlit authentication, Phase 16 added persistent SQLite document storage with per-user duplicate upload protection, and Phase 17B completed Streamlit UI polish, persistent session cookie management, and refined validation/search styling. The remaining work focuses on final real-environment evaluation reruns, screenshots, dissertation/report writing, and presentation preparation.

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
- Phase 17B: Polished Streamlit UI, persistent session cookies, and interactive
  review workflows.

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
- 4b180a7 docs: update project status after phase 16.
- ad75490 Phase 17B: Polish Streamlit UI.

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

- The separate Flask backend was consolidated into a Streamlit-first integrated
  application to simplify local deployment and provide direct state management
  for in-memory FAISS indexing.
- Authentication and database persistence were initially deferred during core ML
  pipeline prototyping, and were subsequently implemented in Phase 15 (auth)
  and Phase 16 (SQLite persistence and duplicate protection), followed by
  Phase 17B session and UI polish.
- MySQL was proposed originally; SQLite is used for local academic evaluation
  and self-contained reproducibility, while maintaining a portable relational
  schema structure.
- BGE-M3 was replaced with `sentence-transformers/all-MiniLM-L6-v2` for
  lightweight, CPU-friendly embedding generation on standard computing resources.
- Additional academic evaluation phases were added beyond the original
  proposal: advisory validation boundary, MRR/NDCG retrieval evaluation,
  stage-level CPU latency benchmarking, and layout-aware classification comparison.

## Remaining Technical / Evaluation Work

- Development tuning and quality improvements where academically justified.
- Final production-environment semantic-search evaluation reruns with real embeddings.
- Final real-document latency benchmark measurements in the full dependency environment.
- Dataset and generalization improvements if feasible.
- Final screenshots and reproducibility checks.
- Dissertation/report writing and final presentation preparation.

## Phase 18: Hybrid Search Reliability Update (17B.4)
- **Status:** Complete
- **Feature:** Deterministic structured query parsing and candidate filtering before semantic ranking.
- **Details:** Resolves issue where pure embeddings fail on numeric inequality, ranges, and hard constraints. Added query_parser.py which extracts conditions like elow 3000, invoice, or exact identifiers, falling back to FAISS for semantic remaining text. Zero-match correctly returns no results. Added conservative relevance threshold support.
