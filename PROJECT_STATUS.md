# Project Status

## Current Status

The core IDP pipeline, user interface, authentication, persistence, and all final empirical evaluation benchmarks are complete. Phase 15 added local Streamlit authentication, Phase 16 added persistent SQLite document storage with per-user duplicate upload protection, Phase 17B/18 added UI polish, persistent session cookies, and hybrid structured semantic search, and Phase 19A refined extraction reliability and search evaluation schemas. All four formal evaluation benchmarks (information extraction, three-class classifier challenge, CPU latency profiling, and 19-query hybrid/semantic search retrieval) are now complete and documented with frozen manifests.

The main implementation and experimental evaluation phase is complete. Remaining work focuses on dissertation/report writing, screenshot capture, reproducibility verification, and presentation preparation.

The current system processes invoices, receipts, and purchase orders through the local pipeline:

```text
Upload document
-> text extraction / OCR
-> document classification
-> field extraction
-> validation and confidence matrix
-> semantic search indexing
-> Streamlit display
```

## Completed Phases

- Phase 1: Established the project structure and active `src/idp_system/` package layout.
- Phase 2: Added document text extraction for local files.
- Phase 3: Added the baseline document classification component.
- Phase 4: Added information extraction for common business-document fields.
- Phase 5: Added semantic search with sentence-transformers and FAISS.
- Phase 6: Integrated the pipeline through the `IDPSystem` orchestrator.
- Phase 7/8: Added the Streamlit UI and classification/extraction heuristic polish.
- Phase 9: Added lightweight evaluation scripts for classification and extraction baselines.
- Phase 10A: Added RVL-CDIP OCR cache builder.
- Phase 10B: Added custom text dataset builders.
- Phase 10C: Added trained real three-class classifier workflow.
- Phase 10D/10E/10F: Added document-type-aware extraction and reliability cleanup.
- Phase 11: Added validation boundary and confidence matrix.
- Phase 11B: Improved receipt and purchase order validation reliability.
- Phase 12: Added semantic search evaluation with Precision@K, Recall@K, MRR@K, and NDCG@K.
- Phase 13: Added CPU latency benchmark framework.
- Phase 14: Added layout-aware classification comparison.
- Phase 15: Added local SQLite-backed Streamlit authentication and user session management.
- Phase 16: Added persistent SQLite document storage and per-user duplicate upload protection.
- Phase 17B: Polished Streamlit UI, persistent session cookies, and interactive review workflows.
- Phase 18: Added deterministic structured query parsing and candidate filtering for hybrid search.
- Phase 19A: Refined evaluation schemas, extraction reliability on POs/invoices, and safe file cleanup.
- Final Empirical Evaluations: Executed, verified, and frozen benchmarks across all four experimental pillars (Extraction V2, Classifier Challenge V2, CPU Latency, and Semantic Search).

## Latest Working Behavior

- Invoice classification and document-type-aware extraction are working.
- Receipt classification and document-type-aware extraction are working.
- Purchase order classification and document-type-aware extraction are working.
- Heuristic overrides improve reliability for strong invoice, receipt, and purchase order signals.
- Validation confidence matrix is working and includes text quality, classification confidence/source, field validation, warning counts, and overall pipeline status.
- Hybrid semantic search combines structured query parsing (amounts, dates, types, exact identifiers/suppliers) with MiniLM embeddings and FAISS ranking.
- Streamlit UI displays uploaded document results, extracted fields, validation details, browser-safe fullscreen PDF preview, paginated history browsing, and search.
- Complete empirical evaluation reports are available in `evaluation/` with frozen manifests and reproducible result sets.

## Recent Commits

- `ca7eea4` Add final semantic search evaluation
- `29c7f7f` Add final CPU latency evaluation
- `43c0a8c` Add final classifier challenge evaluation
- `bb9b4a4` Add final extraction evaluation results
- `f8b80a6` Add frozen final extraction benchmark
- `fe4df9e` Improve receipt extraction reliability
- `5e3d6b6` docs: update Phase 19A evaluation and reliability status
- `9963634` Phase 19A: Improve evaluation and extraction reliability
- `d880ae3` Add paginated document history browsing
- `23bb703` Refine compact authentication page layout
- `f82afe4` Add browser-safe fullscreen document preview
- `c94d9db` Add structured date search filters
- `f4988e5` Return only exact entity matches when available
- `b524148` Document hybrid search behavior
- `a181abf` Improve search matching and Streamlit responsiveness
- `c2a8be3` Phase 18: Add hybrid structured and semantic search
- `eaa4e56` docs: add proposal implementation comparison
- `a308c62` docs: synchronize project documentation after phase 17B
- `d511ec3` Phase 17B.3: Persist authentication and document UI state
- `ad75490` Phase 17B: Polish Streamlit UI

## Known Limitations

- Authentication is implemented as a local SQLite-backed academic prototype. It does not include MFA or RBAC.
- Processed document records persist in SQLite with per-user duplicate protection.
- Semantic search embeddings are in-memory and rebuilt from persisted documents for the signed-in user when needed.
- Validation is advisory only and does not block classification, extraction, or indexing.
- The classifier's reported validation accuracy on the development split is high (~100%), but testing against the independent 42-document challenge set revealed substantial generalization challenges under cross-lingual/image-OCR distribution shifts (see [FINAL_CLASSIFIER_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_CLASSIFIER_RESULTS.md)).
- Receipt training samples far outnumber purchase order samples in the training set.
- Extraction rules rely on explicit keyword anchors and regex heuristics, showing lower recall on out-of-distribution layouts and degraded OCR images (see [FINAL_EXTRACTION_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_EXTRACTION_RESULTS.md)).
- High-resolution scanned image OCR is CPU-intensive, creating the primary pipeline latency bottleneck (see [FINAL_LATENCY_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_LATENCY_RESULTS.md)).

## Changes From Original Proposal

- The separate Flask backend was consolidated into a Streamlit-first integrated application to simplify local deployment and provide direct state management for in-memory FAISS indexing.
- Authentication and database persistence were initially deferred during core ML pipeline prototyping, and were subsequently implemented in Phase 15 (auth) and Phase 16 (SQLite persistence and duplicate protection), followed by Phase 17B session and UI polish.
- MySQL was proposed originally; SQLite is used for local academic evaluation and self-contained reproducibility, while maintaining a portable relational schema structure.
- BGE-M3 was replaced with `sentence-transformers/all-MiniLM-L6-v2` for lightweight, CPU-friendly embedding generation on standard computing resources.
- Additional academic evaluation phases were added beyond the original proposal: advisory validation boundary, MRR/NDCG retrieval evaluation, stage-level CPU latency benchmarking, layout-aware classification comparison, and four frozen final benchmark evaluations.

## Final Empirical Evaluations Summary

The four experimental pillars have been fully executed, verified, and frozen:

1. **Information Extraction Benchmark (V2)**:
   - **Manifest**: `evaluation/final_extraction_benchmark_v2.csv` (63 documents: 29 Invoices, 25 Receipts, 9 POs; SHA-256: `ae7e45d4...`)
   - **Results & Analysis**: See [FINAL_EXTRACTION_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_EXTRACTION_RESULTS.md) (committed at `bb9b4a4`).
   - **Key Finding**: 58.26% overall field exact-match accuracy; 87.41% on native text vs. 10.34% on scanned OCR receipts where recognition degradation dominated.

2. **Three-Class Document Classifier Challenge (V2)**:
   - **Manifest**: `evaluation/final_classifier_challenge_v2.csv` (42 unseen documents; SHA-256: `9c1d629d...`)
   - **Results & Analysis**: See [FINAL_CLASSIFIER_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_CLASSIFIER_RESULTS.md) (committed at `43c0a8c`).
   - **Key Finding**: 64.29% end-to-end pipeline routing accuracy (100% on English native PDFs, 0% on Portuguese OCR receipt images due to confounded domain/language/modality shift).

3. **Stage-by-Stage CPU Latency Benchmark**:
   - **Manifest**: `evaluation/final_latency_benchmark.csv` (29 documents; SHA-256: `eafefb33...`)
   - **Results & Analysis**: See [FINAL_LATENCY_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_LATENCY_RESULTS.md) (committed at `29c7f7f`).
   - **Key Finding**: Native PDFs process with sub-second steady-state latency (mean 0.425s, median 0.248s), while image OCR represents the primary computational bottleneck (mean 133.89s, median 57.64s).

4. **19-Query Semantic Search & Retrieval Benchmark**:
   - **Corpus & Queries**: 6 ground-truth business documents, 19 hybrid and semantic queries.
   - **Results & Analysis**: See [FINAL_SEARCH_RESULTS.md](file:///d:/Campus/Degree/Final%20Project/IDP-System/evaluation/FINAL_SEARCH_RESULTS.md) (committed at `ca7eea4`).
   - **Key Finding**: Overall MRR@5 of 0.9722, NDCG@5 of 0.9574, Semantic-only Recall@5 of 0.9722, and 100% (1/1) out-of-bounds structured query rejection.

## Remaining Work

- Final dissertation and academic report writing.
- Capturing final UI walkthrough screenshots.
- Preparing project demonstration slides and defense presentation.
