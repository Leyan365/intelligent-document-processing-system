# Intelligent Document Processing System

This repository is being migrated into a local-first Intelligent Document
Processing system for final-year project work. The new package lives under
`src/idp_system/` while the old `src/training_data_bot/` package is kept in
place during migration.

Phase 1 sets up the reviewable project structure only. It includes:

- `core/` for configuration, exceptions, logging helpers, and document models.
- `pipeline/` for loader, preprocessing, OCR, classification, extraction,
  embeddings, and semantic-search components.
- `database/` for the future MySQL adapter.
- `system.py` for the top-level `IDPSystem` orchestrator.

The OCR, machine learning, database, semantic search, Flask backend, and
Streamlit dashboard implementations are intentionally left for later phases.

Basic import check:

```powershell
$env:PYTHONPATH="src"; python -c "from idp_system.system import IDPSystem; print(IDPSystem)"
```
