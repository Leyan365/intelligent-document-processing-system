# Final Extraction Benchmark Results

## 1. Benchmark Details
* **Benchmark Identity**: `evaluation/final_extraction_benchmark_v2.csv`
* **Benchmark Hash (SHA-256)**: `ae7e45d4b273002da476c54208c346cc31f5c53b473adc3900da89bf6a9774d9`
* **Git Commit Evaluated**: `f8b80a6` (Add frozen final extraction benchmark)

## 2. Environment
* **Python**: 3.10.19 (Note: Although `requirements.txt` correctly pins scikit-learn to 1.7.1 for classifier artifact compatibility, the evaluation environment strictly used 3.10.19 and scikit-learn 1.7.2. Extraction relies primarily on regexes and OCR, not the classifier).
* **scikit-learn**: 1.7.2
* **PyMuPDF**: 1.26.7
* **PaddleOCR**: 3.5.0
* **PaddlePaddle**: 3.2.0

## 3. Benchmark Composition
* **Total Documents**: 63
  * 29 Invoices (Homogeneous SuperStore, synthetic, native-text)
  * 25 Receipts (SROIE dataset, heterogeneous, scanned images)
  * 9 Purchase Orders (Heterogeneous layouts, native-text)
* **Gold-PRESENT Eligible Fields**: 230
  * Supplier: 59
  * Date: 52
  * Amount: 63
  * Document Number: 56
* **Ambiguous Exclusions**: 14 (11 Dates, 3 Document Numbers)

## 4. Primary Overall Metrics
* **Processing**: 62 completed successfully, 1 OCR receipt processing failure.
* **Overall Eligible-Field Value Accuracy**: 58.26% (134 / 230 fields correct)
  * **Supplier**: 52.54% (31 / 59)
  * **Date**: 75.00% (39 / 52)
  * **Amount**: 55.56% (35 / 63)
  * **Document Number**: 51.79% (29 / 56)
* **Supplier Presence (Precision/Recall/F1)**: 93.22% / 93.22% / 0.932
* **Document-Number Presence (Precision/Recall/F1)**: 100.0% / 55.35% / 0.712

## 5. Results by Class
* **Invoices (29 docs)**: 100.0% accuracy (108 / 108 PRESENT fields correct).
* **Purchase Orders (9 docs)**: 48.57% accuracy (17 / 35 PRESENT fields correct).
* **Receipts (25 docs)**: 10.34% accuracy (9 / 87 PRESENT fields correct).

## 6. Native vs OCR Results
* **Native-Text Documents (38 docs)**: 87.41% overall eligible-field value accuracy.
* **OCR/Scanned Documents (25 docs)**: 10.34% overall eligible-field accuracy.

## 7. NOT_PRESENT Tests
* **Supplier (4 tests)**: 4 false positives (0% correct empty). All four invoices extracted `"Balance Due"` as the supplier due to fallback spatial heuristics capturing the last remaining non-empty line.
* **Document Number (4 tests)**: 0 false positives (100% correct empty).

## 8. Receipt Error Attribution
Of the 78 incorrect receipt fields (from 87 eligible - 9 correct):
* **OCR_RECOGNITION**: 61 (78.2%) — Required characters were missing or misrecognized.
* **EXTRACTION_RULE**: 13 (16.7%) — OCR output contained sufficient correct text, but extraction rules failed to select it.
* **PROCESSING_FAILURE**: 4 (5.1%) — Document could not complete OCR processing.

## 9. PO Error Categories
Of the 18 incorrect PO fields:
* **Unsupported label/pattern**: 10
* **Multiline/layout association**: 6
* **Wrong numeric candidate**: 2

## 10. Validation Effectiveness
* **Incorrect documents that triggered warnings**: 32
* **Incorrect documents missed by validation (false reassurance)**: 5 (These documents yielded a clean `validation_score` and `warnings = []`, showing the heuristic rules failed to detect factual inaccuracy).
* **Clean documents that received warnings**: 0

## 11. Limitations
The extraction rules are highly dependent on explicitly defined keyword anchors (e.g. "Bill To") and rigid layout structures. When applied to out-of-distribution native texts like unseen Purchase Orders, the rules fail due to unfamiliar vocabulary and multiline association errors. On scanned images like the SROIE dataset, OCR degradation further limits the ability of regex rules to find anchors. Furthermore, the supplier fallback rule's tendency to extract false positives in genuinely sparse documents demonstrates a lack of semantic context abstention.

## 12. Interpretation
The final extraction benchmark showed strong performance on the homogeneous native-text SuperStore invoice subset but substantially lower accuracy on independently held-out purchase-order layouts and scanned SROIE receipts. Manual receipt error analysis identified OCR recognition degradation as the primary source of receipt failures, while a smaller but material subset failed despite usable OCR text. The results therefore demonstrate sensitivity to both OCR quality and document-layout variation. Supplier extraction also showed weak abstention on naturally sparse invoice documents. The advisory validation layer identified many problematic outputs but did not detect every factually incorrect extraction.
