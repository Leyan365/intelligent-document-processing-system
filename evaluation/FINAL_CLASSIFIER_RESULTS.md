# Final Three-Class Document Classifier Challenge-Set Evaluation Report

## Executive Summary

This document presents the official, empirical evaluation of the production document classification pipeline on the frozen **42-document independent challenge set** (`evaluation/final_classifier_challenge_v2.csv`).

The evaluation was conducted as a one-time frozen benchmark without retraining the classifier, modifying TF-IDF parameters, adjusting Logistic Regression weights, tuning heuristic overrides, or altering the challenge manifest.

### High-Level Performance Comparison

| Metric Layer | Evaluation Basis | Production Pipeline (With Heuristics) | Raw ML Model (TF-IDF + Logistic Regression) |
| :--- | :--- | :---: | :---: |
| **End-to-End Pipeline Routing Accuracy** | All 42 Challenge Documents (Treats extraction failures as unsuccessful) | **64.29%** (27 / 42) | **57.14%** (24 / 42) |
| **Classifier Accuracy Conditional on Successful Text Extraction** | 39 Successfully Extracted Documents (Evaluates purely classifier decision logic) | **69.23%** (27 / 39) | **61.54%** (24 / 39) |
| **Macro F1-Score (End-to-End)** | 42 Documents (3 Target Classes) | **0.5833** | **0.5230** |
| **Weighted F1-Score (End-to-End)** | 42 Documents (3 Target Classes) | **0.5357** | **0.4849** |
| **English Sub-Set Accuracy** | 27 Documents (18 Invoices, 9 POs) | **100.0%** (27 / 27) | **88.89%** (24 / 27) |
| **Portuguese Sub-Set Accuracy** | 15 Documents (15 Receipts from NOVA IMS) | **0.0%** (0 / 15) | **0.0%** (0 / 15) |
| **Native PDF Text Accuracy** | 27 Documents (Native digital PDFs) | **100.0%** (27 / 27) | **88.89%** (24 / 27) |
| **OCR Image Accuracy** | 15 Documents (Scanned / photographed images) | **0.0%** (0 / 15) | **0.0%** (0 / 15) |

---

## 1. Challenge Manifest & Freeze Verification

- **Manifest Path:** `evaluation/final_classifier_challenge_v2.csv`
- **Manifest SHA-256:** `9c1d629d9a9c32c85c50da7bdc68e81503061ca8d5caf5b886294959f02a90cc`
- **V1 Manifest SHA-256:** `04bbbfe51f795af73c516e2344fd8dd27dbbec6e067ca38d404e53b22e010be8` (Preserved unchanged as historical evidence)
- **Total Challenge Documents:** 42
- **Historical Status:** 100% `FINAL_UNSEEN`
- **Contamination Check:** Re-verified against 5,766 historical project file hashes (0 overlap).

---

## 2. Runtime Environment & Classifier Artifact Details

- **Git Commit Evaluated:** `bb9b4a4a3dbb21e7fba241d3fd086ee462f71876`
- **Python Version:** 3.10.19 (Anaconda `dsenv`)
- **scikit-learn Version:** 1.7.2 (Runtime) / 1.7.1 (Artifact serialization version)
- **PyMuPDF Version:** 1.26.7
- **PaddleOCR Version:** 3.5.0
- **PaddlePaddle Version:** 3.2.0
- **Classifier Artifact Path:** `models/document_classifier.joblib`
- **Pipeline Architecture:** `TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english', lowercase=True)` $\rightarrow$ `LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)`
- **Target Classes:** `['invoice', 'purchase_order', 'receipt']`
- **Version Compatibility:** An `InconsistentVersionWarning` was recorded during model unpickling due to the minor patch difference between scikit-learn 1.7.1 and 1.7.2. Model execution was verified without numerical degradation.

---

## 3. Detailed Performance Metrics

### A. End-to-End Pipeline Routing Performance (All 42 Documents)
Evaluated across all 42 challenge documents; treats the 3 text extraction failures (`FC-REC-007`, `FC-REC-009`, `FC-REC-010`) as unsuccessful routing outcomes (false negatives for the `receipt` class).

| Document Class | Gold Support | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Invoice** | 18 | 18 | 12 | 0 | **0.6000** | **1.0000** | **0.7500** |
| **Receipt** | 15 | 0 | 0 | 15 | **0.0000** | **0.0000** | **0.0000** |
| **Purchase Order** | 9 | 9 | 0 | 0 | **1.0000** | **1.0000** | **1.0000** |
| **Macro Average (3 Classes)** | **42** | — | — | — | **0.5333** | **0.6667** | **0.5833** |
| **Weighted Average** | **42** | — | — | — | **0.4714** | **0.6429** | **0.5357** |

### B. Classifier Performance Conditional on Successful Text Extraction (39 Documents)
Evaluated strictly across the 39 documents where text extraction succeeded, measuring pure classifier decision accuracy.

| Document Class | Gold Support | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Invoice** | 18 | 18 | 12 | 0 | **0.6000** | **1.0000** | **0.7500** |
| **Receipt** | 12 | 0 | 0 | 12 | **0.0000** | **0.0000** | **0.0000** |
| **Purchase Order** | 9 | 9 | 0 | 0 | **1.0000** | **1.0000** | **1.0000** |
| **Macro Average (3 Classes)** | **39** | — | — | — | **0.5333** | **0.6667** | **0.5833** |
| **Weighted Average** | **39** | — | — | — | **0.5077** | **0.6923** | **0.5769** |

---

## 4. Confusion Matrices

### A. Production Pipeline-Routing Confusion Matrix (All 42 Documents)
This matrix accounts for all 42 challenge documents across the three gold classes and all four pipeline routing outcomes:

| Gold Class \ Pipeline Output | Predicted Invoice | Predicted Receipt | Predicted Purchase Order | EXTRACTION_FAILED | Total Gold Documents |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Invoice** | **18** | 0 | 0 | 0 | **18** |
| **Receipt** | **12** | **0** | 0 | 3 | **15** |
| **Purchase Order** | 0 | 0 | **9** | 0 | **9** |
| **Total Predicted** | **30** | **0** | **9** | **3** | **42** |

### B. Classifier-Only Confusion Matrix (39 Text-Extracted Documents)
When evaluating strictly the classifier module on documents where OCR/text extraction succeeded:

| Gold Class \ Classifier Prediction | Predicted Invoice | Predicted Receipt | Predicted Purchase Order | Total Evaluated |
| :--- | :---: | :---: | :---: | :---: |
| **Invoice** | **18** | 0 | 0 | **18** |
| **Receipt** | **12** | **0** | 0 | **12** |
| **Purchase Order** | 0 | 0 | **9** | **9** |
| **Total Predicted** | **30** | **0** | **9** | **39** |

*(Note: The 3 receipt documents with preprocessing failures are excluded from Table B and accounted for in Table A).*

---

## 5. Confounding Analysis & Scope Limitations

> [!WARNING]
> **Complete Statistical Confounding:** In this challenge benchmark:
> - All 15 receipt documents are Portuguese-language, photographed/scanned image files (`ocr_image`), belonging to the `receipt` class, and sourced from the NOVA IMS dataset.
> - All 27 invoice and purchase-order documents are English-language, digital PDFs (`native_pdf_text`), belonging to the `invoice` or `purchase_order` class.
>
> Consequently, this challenge benchmark cannot statistically isolate the independent causal contributions of **language**, **modality**, **document class**, or **source-domain shifts**.

### Specific Subset Results:
- **English Invoices & Purchase Orders (27 documents):** The production classifier correctly classified all 27 independently held-out English invoice and purchase-order documents in this challenge set.
  - *Limitations:* The 18 invoices originate from a single SuperStore synthetic template family, and the 9 purchase orders represent a limited set of industrial suppliers. These results demonstrate robustness on these held-out templates but should not be extrapolated to unconstrained commercial invoice/PO populations.
- **Portuguese Receipts (15 documents):** None of the 15 receipts were correctly routed to `receipt` (12 routed to `invoice`, 3 failed during OCR preprocessing).

---

## 6. Receipt Diagnostic Analysis

A rule-based diagnostic heuristic was applied to inspect the extracted OCR text of the 15 receipt images:

| Diagnostic Category | Count | Diagnostic Criteria / Context |
| :--- | :---: | :---: |
| **CROSS_LINGUAL_VOCABULARY (Diagnostic Heuristic)** | 8 | Clean OCR text containing standard Portuguese POS terminology (*Fatura Simplificada, Total, IVA, Artigo, Caixa, Dinheiro*). The English TF-IDF model lacked Portuguese n-grams, and English regex rules (*"receipt", "cashier"*) did not match. The occurrence of *"Fatura"* biased the bag-of-words model toward `invoice`. |
| **CLASSIFIER_GENERALIZATION (Diagnostic Heuristic)** | 4 | Text was extracted, but lack of spatial/geometric feature modeling prevented the model from recognizing receipt layouts. |
| **PROCESSING_FAILURE** | 3 | PaddleOCR raised runtime exceptions on 3 high-resolution/rotated images (`FC-REC-007`, `FC-REC-009`, `FC-REC-010`). |
| **OCR_DEGRADATION** | 0 | Severe text degradation was not the primary driver where OCR extraction succeeded. |

*Methodological Caveat:* The presence of Portuguese keywords demonstrates that usable text was recovered by OCR, but this diagnostic does not prove that language shift alone caused the misclassifications, given the simultaneous presence of modality and template confounding.

---

## 7. Invoice & Purchase Order Analysis (Heuristic Effects)

- **Correct Classifications (27 / 27):** All 18 invoices and 9 purchase orders were correctly classified in the final production pipeline.
- **Corrective Heuristic Routing:**
  - `FC-INV-001` (sparse zero-total invoice): The raw ML model predicted `receipt`, but production invoice heuristics correctly routed it to `invoice`.
  - `FC-PO-004` & `FC-PO-005` (trim purchase orders): The raw ML model predicted `invoice` due to shared supplier/tax terminology, but production purchase-order heuristics correctly routed them to `purchase_order`.
- **Heuristic Impact Summary:** Production heuristics triggered on 27 / 42 documents, correcting 3 raw-model misclassifications and raising end-to-end routing accuracy from 57.14% to 64.29% (and conditional accuracy from 61.54% to 69.23%). Heuristics did not trigger on any Portuguese receipts because all regex rules require English keywords.

---

## 8. Model Confidence & Probability Analysis

For documents where raw ML model prediction probabilities were evaluated:
- **Mean Confidence for Correct Predictions:** `0.6628` (66.28%)
- **Mean Confidence for Incorrect Predictions:** `0.6448` (64.48%)
- **Lowest Confidence Correct Prediction:** `0.4541` (45.41%) on `FC-PO-009`
- **Highest Confidence Incorrect Prediction:** `0.8372` (83.72%) on `FC-REC-012` (misclassified as invoice)

*Interpretation:* Model confidence did not strongly separate correct from incorrect predictions on this challenge set, and at least one incorrect receipt prediction received high confidence (0.8372). These output probabilities should not be treated as calibrated confidence measures.

---

## 9. Dissertation-Safe Interpretation

The independent challenge evaluation produced **64.3% end-to-end production routing accuracy** (and **69.2% classifier accuracy conditional on successful text extraction**), compared with **57.1% end-to-end raw-model accuracy** (**61.5% conditional** for the raw TF-IDF / Logistic Regression model). Heuristic routing corrected three raw-model errors in the evaluated set. All 18 invoice and nine purchase-order documents were correctly routed by the production classifier, whereas none of the 15 Portuguese OCR receipt documents were correctly classified; three of these receipt images failed during OCR preprocessing before classification.

The receipt result demonstrates a substantial generalisation limitation under a combined cross-domain, cross-lingual and image/OCR distribution shift. Because receipt class, Portuguese language, OCR modality and NOVA source are perfectly confounded in this challenge set, their individual causal contributions cannot be isolated. Diagnostic inspection indicates that Portuguese transactional text was recoverable for several misclassified receipts, suggesting vocabulary/domain shift as one contributing factor rather than establishing it as the sole cause.

The challenge therefore complements rather than replaces the original source-aligned validation result. The original validation measures performance within the development dataset structure (~100%), whereas the challenge set provides a harder test of distributional generalisation across independently held-out templates and out-of-domain public data.

---

## 10. Summary of Evaluation Artifacts

1. `evaluation/results/final_classifier_v2_predictions.csv` — Document-level predictions and metadata.
2. `evaluation/results/final_classifier_v2_metrics.json` — End-to-end and conditional precision, recall, F1, and accuracy metrics.
3. `evaluation/results/final_classifier_v2_confusion_matrix.csv` — 3x4 Pipeline-Routing and 3x3 Classifier-Only confusion matrices.
4. `evaluation/results/final_classifier_v2_confusion_matrix.png` — Visual pipeline-routing confusion matrix heatmap.
5. `evaluation/results/final_classifier_v2_summary.json` — Machine-readable evaluation summary.
