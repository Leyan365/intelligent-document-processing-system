# Final CPU Latency Evaluation Report

**Document Processing Pipeline CPU Latency Benchmark**
**Dissertation Experimental Evaluation Evidence**

---

## 1. Executive Summary

- **Benchmark Manifest SHA-256**: `eafefb33c3f464613b27ee55c7612c34d0acda16bb303b3b8865dc96fca884d5`
- **Git Commit Evaluated**: `43c0a8c7c73d5b9de07e07b2c6b3119cd8a62c5a`
- **Hardware Environment**: 13th Gen Intel(R) Core(TM) i5-13420H (8 physical cores / 12 logical threads), 15.67 GB RAM, Battery (52%), None (strictly CPU-only).
- **Sample Size**: 29 benchmark documents (10 Invoices, 9 Purchase Orders, 10 Receipts; 19 Native PDFs, 10 OCR Images).
- **Repetition Policy**: Native PDFs evaluated with 3 measured repetitions per document; OCR receipts evaluated with 1 measured repetition per document (Total: 67 measured steady-state attempts; 66 successful runs across 28 unique successful documents).
- **Document-Balanced Steady-State Latency** (N=28 successful documents, 1 value per document): Mean = **43.3260s**, Median = **0.3331s**, Min = **0.1061s**, Max = **725.0293s**, P90 = **68.1587s** (Std Dev = 136.3995s). *This document-balanced metric is the preferred headline metric to avoid repetition bias between native PDFs (3 reps) and OCR receipts (1 rep).*
- **Run-Weighted Steady-State Latency** (N=66 successful measured runs): Mean = **18.6255s**, Median = **0.2846s**, Min = **0.0714s**, Max = **725.0293s** (Std Dev = 90.4698s).
- **Native Text Processing** (N=19 docs): Mean = **0.4250s**, Median = **0.2480s**, Min = **0.1061s**, Max = **1.7884s** (Std Dev = 0.4944s).
- **OCR Image Processing** (N=9 successful docs): Mean = **133.8948s**, Median = **57.6401s**, Min = **54.0775s**, Max = **725.0293s** (Std Dev = 221.7916s). *Note: The OCR mean (133.89s) is strongly affected by a single extreme 725.03s observation (LAT-REC-006); the median (57.64s) is more representative of typical successful OCR performance for this small sample.*
- **Modality Latency Ratio**: OCR documents required approximately **315.02x** the processing time of native-text documents on CPU (median ratio: **232.4x**).
- **Cold-Start Initialization Cost** (Same-Document Comparison):
  - Native PDF (`LAT-INV-001`): Cold first run = **5.5665s** vs same-document steady-state mean = **0.1061s** (Overhead: **+5.4603s**, **52.46x**).
  - OCR Receipt (`LAT-REC-001`): Cold first run = **237.2120s** vs same-document steady-state = **75.3861s** (Overhead: **+161.8259s**, **3.15x**).
- **Processing Success Rate**: **96.55%** (28/29 documents successfully processed; 1 OCR image failure isolated and logged separately without biasing successful distribution metrics).

---

## 2. Experimental Environment & Hardware Setup

| Parameter | Value |
| :--- | :--- |
| **Operating System** | Windows-10-10.0.26200-SP0 |
| **CPU Model** | 13th Gen Intel(R) Core(TM) i5-13420H |
| **Physical / Logical Cores** | 8 Physical / 12 Logical |
| **Total RAM / Available** | 15.67 GB / 5.07 GB |
| **Power Mode** | Battery (52%) |
| **GPU Acceleration** | None (strictly CPU-only) |
| **Paddle Device** | cpu (CUDA compiled: False) |
| **Python Version** | 3.10.19 |
| **scikit-learn Version** | 1.7.2 |
| **PyMuPDF Version** | 1.26.7 |
| **PaddleOCR / PaddlePaddle** | 3.5.0 / 3.2.0 |
| **sentence-transformers** | 5.4.1 (`sentence-transformers/all-MiniLM-L6-v2`) |
| **FAISS Version** | 1.13.2 |
| **OpenCV / NumPy** | 4.10.0 / 1.26.4 |
| **Initial / Peak RSS Memory** | 588.68 MB / 1300.85 MB |
| **Peak Working Set** | 8076.94 MB |
| **Private Memory** | 15048.62 MB |

> **Note on Resource Observations**: Initial RSS, peak RSS, peak working set, and private memory are descriptive process-level observations captured at different points during the evaluation; they do not constitute a controlled memory-leak series.
> **Note on Benchmark Operating Conditions**: The benchmark executed on battery power (52% remaining) with background CPU utilization around 18–20%. Absolute timing figures are specific to this runtime environment and should not be generalized as universal throughput across different hardware or thermal configurations.

---

## 3. End-to-End Steady-State Latency Distributions

### 3.1 Document-Balanced Steady-State Latency (1 Value per Document, N=28)

| Group / Class | Modality | N (docs) | Mean (s) | Median (s) | Std Dev (s) | Min (s) | Max (s) | P90 (s) | IQR (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Documents (Overall)** | Mixed | 28 | **43.3260** | **0.3331** | 136.3995 | 0.1061 | 725.0293 | 68.1587 | 53.9161 |
| **Native Text PDFs** | Native PyMuPDF | 19 | **0.4250** | **0.2480** | 0.4944 | 0.1061 | 1.7884 | 1.4002 | 0.1884 |
| **Scanned/Image Receipts** | PaddleOCR + CV | 9 | **133.8948** | **57.6401** | 221.7916 | 54.0775 | 725.0293 | 725.0293 | 12.7997 |
| `invoice` | Native PDF | 10 | 0.1617 | 0.1540 | 0.0393 | 0.1061 | 0.2480 | 0.1974 | 0.0381 |
| `purchase_order` | Native PDF | 9 | 0.7176 | 0.3367 | 0.6044 | 0.2514 | 1.7884 | 1.7884 | 1.0261 |
| `receipt` | OCR Image | 9 | 133.8948 | 57.6401 | 221.7916 | 54.0775 | 725.0293 | 725.0293 | 12.7997 |

### 3.2 Run-Weighted Steady-State Latency (All Successful Measured Runs, N=66)

| Group | N (runs) | Mean (s) | Median (s) | Std Dev (s) | Min (s) | Max (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Runs** | 66 | **18.6255** | **0.2846** | 90.4698 | 0.0714 | 725.0293 |
| **Native PDF Runs** | 57 | **0.4250** | **0.2283** | 0.4941 | 0.0714 | 1.9149 |
| **OCR Image Runs** | 9 | **133.8948** | **57.6401** | 221.7916 | 54.0775 | 725.0293 |

> **Note on Percentiles**: Percentiles (P90, P95, IQR) are calculated using linear interpolation on sorted observations. With N=9 successful OCR documents, P90 is heavily weighted by the maximum observation and should not be over-interpreted.

---

## 4. Stage-Level Latency Breakdown & Computational Bottlenecks

### 4.1 Native Text Documents (Invoices & Purchase Orders)

| Processing Stage | Implementation Component | Mean Latency (s) | Contribution (%) |
| :--- | :--- | :---: | :---: |
| **1. File Load & Direct Extraction** | PyMuPDF (`_extract_pdf_text_with_pymupdf`) | 0.029219s | 6.87% |
| **2. Text Normalization** | Regex text cleaner (`clean_text`) | 0.000301s | 0.07% |
| **3. Document Classification** | TF-IDF + Logistic Regression | 0.000229s | 0.05% |
| **4. Information Extraction** | Regex / Token Rule Extractor | 0.291562s | 68.60% |
| **5. Pipeline Validation** | Multi-rule cross-validator | 0.001645s | 0.39% |
| **6. Embedding Generation** | `all-MiniLM-L6-v2` (SentenceTransformer) | 0.101881s | 23.97% |
| **7. Semantic Index Update** | FAISS index insertion | 0.000064s | 0.02% |
| **Total End-to-End** | Complete Pipeline | **0.425033s** | **100.00%** |
| *Downstream Stages Total* | *Stages 3 to 7 combined* | *0.395381s* | *93.02%* |

> **Dominant Stage for Native PDFs**: `field_extraction_time_s` (0.291562s, **68.60%** of total runtime).

### 4.2 OCR / Image Documents (Retail Receipts)

| Processing Stage | Implementation Component | Mean Latency (s) | Contribution (%) |
| :--- | :--- | :---: | :---: |
| **1. Image File Load** | OpenCV `imread` | 0.117315s | 0.09% |
| **2. Image Preprocessing** | OpenCV (Grayscale, Denoise, Adaptive Thresh, Deskew) | 0.204600s | 0.15% |
| **3. OCR Inference** | PaddleOCR (Detection + Angle Cls + Recognition) | 133.099097s | 99.41% |
| **4. OCR Text Assembly** | Text collector & cleaner | 0.000198s | 0.00% |
| **5. Document Classification** | TF-IDF + Logistic Regression | 0.006524s | 0.00% |
| **6. Information Extraction** | Receipt heuristic & anchor extractor | 0.126715s | 0.09% |
| **7. Pipeline Validation** | Multi-rule cross-validator | 0.000817s | 0.00% |
| **8. Embedding Generation** | `all-MiniLM-L6-v2` (SentenceTransformer) | 0.222146s | 0.17% |
| **9. Semantic Index Update** | FAISS index insertion | 0.000137s | 0.00% |
| **Total End-to-End** | Complete Pipeline | **133.894816s** | **100.00%** |
| *Downstream Stages Total* | *Stages 5 to 9 combined* | *0.356338s* | *0.27%* |

> **Dominant Stage for OCR Images**: `ocr_inference_time_s` (133.099097s, **99.41%** of total runtime).

> **Note on Downstream Pipeline Latency**: Once text is available, the remaining classification, extraction, validation, embedding and indexing stages remain sub-second in this benchmark (mean 0.395s for native PDFs, 0.356s for OCR receipts).

> **Note on FAISS Index Configuration**: The evaluation harness utilized an evaluation-local `IndexFlatL2` index during benchmarking, whereas the production semantic search service (`src/idp_system/pipeline/search.py`) utilizes `IndexFlatIP` with normalized embeddings. Vector index addition required only ~0.000064s for native PDFs and ~0.000137s for OCR receipts (<=0.02% of total runtime); this difference does not materially affect the end-to-end timing conclusions.

---

## 5. Cold-Start vs Steady-State Performance

### 5.1 Same-Document Cold-Start Comparison (Primary Baseline)

| Document Modality | Document ID & Filename | Cold First Run (s) | Same-Doc Steady-State Latency (s) | Cold Overhead (s) | Cold Factor | Primary Cold-Start Driver |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Native PDF** | `LAT-INV-001` (`invoice_Amy Hunt_37363.pdf`) | 5.5665s | 0.1061s | +5.4603s | 52.46x | SentenceTransformer model loading & initialization overhead |
| **OCR Receipt** | `LAT-REC-001` (`20210323_181451.jpg`) | 237.2120s | 75.3861s | +161.8259s | 3.15x | PaddleOCR deep neural network weights loading |

### 5.2 Whole-Modality Comparison (Secondary Reference)

| Document Modality | Cold First Run (s) | Whole-Modality Steady Mean (s) | Overhead vs Modality Mean (s) |
| :--- | :---: | :---: | :---: |
| **Native PDF** (`LAT-INV-001`) | 5.5665s | 0.4250s | +5.1414s |
| **OCR Receipt** (`LAT-REC-001`) | 237.2120s | 133.8948s | +103.3172s |

> **Cold-Start Analysis**: The first native run showed additional initialization cost in both the field extraction and embedding stages. SentenceTransformer initialization is visible in the embedding timing, while the benchmark does not isolate the exact source of the additional first-run extraction cost. PaddleOCR initialization accounts for the cold-start overhead in image documents.

---

## 6. Per-Document Benchmark Results

| Latency ID | Filename | Type | Modality | Size (Bytes) | Mean Total (s) | Load (s) | OCR / Text (s) | Classify (s) | Extract (s) | Embed (s) | Index (s) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `LAT-INV-001` | `invoice_Amy Hunt_37363.pdf` | invoice | native_pdf_text | 9,834 | **0.1061** | 0.0229 | 0.0001 | 0.0001 | 0.0345 | 0.0478 | 0.0001 |
| `LAT-INV-002` | `invoice_Bryan Spruell_46338.pdf` | invoice | native_pdf_text | 14,512 | **0.1694** | 0.0056 | 0.0000 | 0.0001 | 0.0556 | 0.1071 | 0.0001 |
| `LAT-INV-003` | `invoice_Doug O'Connell_18233.pdf` | invoice | native_pdf_text | 14,776 | **0.1727** | 0.0099 | 0.0001 | 0.0001 | 0.0649 | 0.0972 | 0.0001 |
| `LAT-INV-004` | `invoice_Sean O'Donnell_10370.pdf` | invoice | native_pdf_text | 14,667 | **0.1483** | 0.0070 | 0.0001 | 0.0001 | 0.0609 | 0.0793 | 0.0001 |
| `LAT-INV-005` | `invoice_Jason Klamczynski_21586.pdf` | invoice | native_pdf_text | 14,805 | **0.1596** | 0.0145 | 0.0000 | 0.0001 | 0.0478 | 0.0968 | 0.0001 |
| `LAT-INV-006` | `invoice_Sarah Brown_4375.pdf` | invoice | native_pdf_text | 14,815 | **0.1974** | 0.0084 | 0.0000 | 0.0001 | 0.0562 | 0.1321 | 0.0001 |
| `LAT-INV-007` | `invoice_Aimee Bixby_39796.pdf` | invoice | native_pdf_text | 15,522 | **0.1466** | 0.0050 | 0.0000 | 0.0001 | 0.0576 | 0.0832 | 0.0001 |
| `LAT-INV-008` | `invoice_Adrian Barton_35580.pdf` | invoice | native_pdf_text | 14,947 | **0.1346** | 0.0039 | 0.0000 | 0.0001 | 0.0465 | 0.0837 | 0.0001 |
| `LAT-INV-009` | `invoice_Harold Engle_3942.pdf` | invoice | native_pdf_text | 15,844 | **0.1344** | 0.0086 | 0.0000 | 0.0001 | 0.0468 | 0.0784 | 0.0001 |
| `LAT-INV-010` | `invoice_Arthur Wiediger_33504.pdf` | invoice | native_pdf_text | 16,758 | **0.2480** | 0.0056 | 0.0001 | 0.0001 | 0.0939 | 0.1477 | 0.0001 |
| `LAT-PO-001` | `PO10049686.pdf` | purchase_order | native_pdf_text | 87,307 | **1.3146** | 0.0312 | 0.0025 | 0.0007 | 1.1601 | 0.1139 | 0.0001 |
| `LAT-PO-002` | `KPO00146851.pdf` | purchase_order | native_pdf_text | 420,562 | **0.2514** | 0.0286 | 0.0002 | 0.0001 | 0.1495 | 0.0722 | 0.0000 |
| `LAT-PO-003` | `PO No - 446161.pdf` | purchase_order | native_pdf_text | 319,866 | **0.4777** | 0.1048 | 0.0004 | 0.0009 | 0.2772 | 0.0928 | 0.0001 |
| `LAT-PO-004` | `RWB-TE1359  -B6Z-B9F-F3E-F66-U10-001-023-782---Heat seal- PO No - 441496....pdf` | purchase_order | native_pdf_text | 530,392 | **1.7884** | 0.1310 | 0.0009 | 0.0008 | 1.5208 | 0.1283 | 0.0001 |
| `LAT-PO-005` | `RWB-TE1432 -B9F-C87-U10-001-023--Heat seal- PO No - 441497.pdf` | purchase_order | native_pdf_text | 454,281 | **1.4002** | 0.1176 | 0.0006 | 0.0005 | 1.1597 | 0.1147 | 0.0001 |
| `LAT-PO-006` | `INPO000006 SCREENLINE SEMEX P.H.P.U FERTACZ #PO - 032026 FLEX TS.PDF` | purchase_order | native_pdf_text | 183,880 | **0.2886** | 0.0127 | 0.0002 | 0.0001 | 0.1659 | 0.1080 | 0.0001 |
| `LAT-PO-007` | `1000598004.pdf` | purchase_order | native_pdf_text | 231,567 | **0.2713** | 0.0099 | 0.0001 | 0.0001 | 0.1394 | 0.1207 | 0.0001 |
| `LAT-PO-008` | `1000600798.pdf` | purchase_order | native_pdf_text | 248,420 | **0.3295** | 0.0174 | 0.0002 | 0.0002 | 0.2100 | 0.1008 | 0.0001 |
| `LAT-PO-009` | `PO#1000601852.pdf` | purchase_order | native_pdf_text | 246,986 | **0.3367** | 0.0109 | 0.0001 | 0.0002 | 0.1924 | 0.1311 | 0.0001 |
| `LAT-REC-001` | `20210323_181451.jpg` | receipt | ocr_image | 639,413 | **75.3861** | 0.1028 | 74.6557 | 0.0032 | 0.0677 | 0.1090 | 0.0001 |
| `LAT-REC-002` | `20210322_161137.jpg` | receipt | ocr_image | 588,768 | **54.0855** | 0.0999 | 53.5080 | 0.0053 | 0.1147 | 0.1294 | 0.0001 |
| `LAT-REC-003` | `20210320_164056.jpg` | receipt | ocr_image | 601,124 | **58.2082** | 0.1153 | 57.5995 | 0.0044 | 0.0765 | 0.1207 | 0.0001 |
| `LAT-REC-004` | `20210428_191514.jpg` | receipt | ocr_image | 469,391 | **55.3589** | 0.1239 | 54.7920 | 0.0040 | 0.0642 | 0.1286 | 0.0003 |
| `LAT-REC-005` | `20210323_171146.jpg` | receipt | ocr_image | 934,639 | **68.1587** | 0.1140 | 66.8478 | 0.0112 | 0.3800 | 0.5659 | 0.0002 |
| `LAT-REC-006` | `20210507_173625.jpg` | receipt | ocr_image | 442,921 | **725.0293** | 0.2603 | 723.2001 | 0.0104 | 0.1634 | 0.6255 | 0.0002 |
| `LAT-REC-007` | `20210323_170725.jpg` | receipt | ocr_image | 594,004 | **FAILED** | NA | NA | NA | NA | NA | NA |
| `LAT-REC-008` | `20210428_194831.jpg` | receipt | ocr_image | 535,218 | **57.6401** | 0.0792 | 57.0581 | 0.0087 | 0.1339 | 0.1217 | 0.0002 |
| `LAT-REC-009` | `20210428_153616.jpg` | receipt | ocr_image | 454,222 | **54.0775** | 0.0940 | 53.5436 | 0.0065 | 0.0956 | 0.1162 | 0.0001 |
| `LAT-REC-010` | `20210316_223932.jpg` | receipt | ocr_image | 330,052 | **57.1091** | 0.0664 | 56.6869 | 0.0052 | 0.0444 | 0.0824 | 0.0001 |

### 6.1 Processing Failure Log

| Latency ID | Filename | Run Number | Stage | Directly Observed Exception |
| :--- | :--- | :---: | :--- | :--- |
| `LAT-REC-007` | `20210323_170725.jpg` | Rep 1 | OCR Inference | `RuntimeError: Unknown exception` |

---

## 7. Dissertation Interpretation & Discussion

### 7.1 Key Findings
1. **Native Text Processing Efficiency**: The tested native-text PDFs were processed in sub-second steady-state latency on this hardware (mean **0.4250 seconds**, median **0.2480 seconds**), supporting interactive local processing for this sample.
2. **Computational OCR Bottleneck**: Successful OCR receipts had a median latency of approximately **57.6401 seconds** (mean **133.8948 seconds**), while one extreme run reached approximately **725.0293 seconds** and one document failed (`LAT-REC-007`). OCR therefore represents the primary CPU latency bottleneck in this benchmark.
3. **Downstream Pipeline Execution**: Once text is available, the remaining classification, extraction, validation, embedding and indexing stages remain sub-second in this benchmark (mean **0.3954s** for native PDFs, **0.3563s** for OCR receipts).
4. **Cold Start vs Steady State**: Cold-start initialization imposes a noticeable first-run penalty primarily driven by model loading and runtime initialization into system memory. Once warmed, subsequent processing proceeds with consistent steady-state timings.

### 7.2 Practical Deployment Recommendations
- **Hybrid Processing Model**: In production deployment, documents should always attempt fast native text extraction first before falling back to OCR only when text content is missing or insufficient.
- **Asynchronous / Background Queuing for OCR**: Because high-resolution photo OCR requires significant CPU time, interactive user interfaces should handle image OCR via asynchronous task queues with visual progress indicators rather than synchronous blocking requests.
- **Pre-Warming Pipeline Services**: Pre-warming can shift initialization cost to application startup and reduce first-request latency.
- **Exploratory Optimization Avenues**: For future deployment, asynchronous processing, model pre-warming, lower-resolution input strategies, or hardware/runtime acceleration could be investigated.

---

*Report generated automatically by `evaluation/final_latency_eval.py`*
