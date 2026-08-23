# Final Semantic Search & Information Retrieval Benchmark Report

**19-Query Hybrid and Semantic Search Evaluation**
**Dissertation Experimental Evidence**

---

## 1. Executive Summary

- **Benchmark Status**: FROZEN 19-Query Business Document Search Evaluation Benchmark
- **Git Commit Evaluated**: `29c7f7ffad8b101a9918c8ccceb9d5b553c55cad`
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **FAISS Index Type**: `faiss.IndexFlatIP` (Exact Inner Product / Cosine Similarity)
- **Benchmark Scale**: 19 Total Queries (18 Scored Relevance Queries, 1 Unmatched Filter Boundary Query)
- **Corpus Scale**: 6 Ground-Truth Business Documents (2 Invoices, 2 Receipts, 2 Purchase Orders)
- **Cutoff Depths (K)**: K = 1, 3, 5 (Retrieval depth top_k = 5)
- **No-Match Query Accuracy**: **1/1** (100.0% correct rejection for out-of-range structured query)

### High-Level Metric Summary

| Metric Cutoff | Overall Precision@K | Overall Recall@K | Overall MRR@K | Overall NDCG@K | Semantic-Only Precision@K | Semantic-Only Recall@K | Semantic-Only MRR@K | Semantic-Only NDCG@K |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **K=1** | **0.9444** | **0.5926** | **0.9444** | **0.9444** | **0.8333** | **0.5278** | **0.8333** | **0.8333** |
| **K=3** | **0.4815** | **0.8333** | **0.9722** | **0.9493** | **0.5000** | **0.8333** | **0.9167** | **0.9030** |
| **K=5** | **0.3222** | **0.8796** | **0.9722** | **0.9574** | **0.4000** | **0.9722** | **0.9167** | **0.9272** |

---

## 2. Experimental Environment & Subsystem Architecture

| Component / Parameter | Specification |
| :--- | :--- |
| **Operating System** | Windows-10-10.0.26200-SP0 |
| **Python Runtime** | 3.10.19 |
| **Embedding Backend** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Dense Vector Dimension** | 384 |
| **Vector Index Backend** | `faiss IndexFlatIP` |
| **sentence-transformers Version** | 5.4.1 |
| **FAISS Version** | 1.13.2 |
| **PyTorch Version** | 2.11.0+cpu |
| **NumPy Version** | 1.26.4 |
| **Search Execution Architecture** | Hybrid: Pre-filtering (Type/Supplier/Amount/Date) $\rightarrow$ 5-Tier Lexical Match $\rightarrow$ Dense FAISS Inner Product Ranking |

---

## 3. Evaluation Methodology & Metric Definitions

The evaluation harness assesses information retrieval quality across 18 scored queries with explicitly defined graded relevance ground truth (grades 1 to 3) and 1 negative filter query.

1. **Precision@K**: The proportion of retrieved documents in the top $K$ results that are relevant:
   $$\text{Precision}@K = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{K}$$
2. **Recall@K**: The proportion of all relevant documents for the query that appear in the top $K$ results:
   $$\text{Recall}@K = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{|\text{Relevant}|}$$
3. **Mean Reciprocal Rank (MRR@K)**: The reciprocal rank of the first relevant document appearing within the top $K$ results (set to 0 if no relevant document appears within rank $K$):
   $$\text{RR}@K = \begin{cases} \frac{1}{\text{rank}_1} & \text{if } \text{rank}_1 \le K \\ 0 & \text{otherwise} \end{cases}, \quad \text{MRR}@K = \frac{1}{|Q|} \sum_{q \in Q} \text{RR}_q@K$$
4. **Normalized Discounted Cumulative Gain (NDCG@K)**: Measures graded ranking quality with exponential relevance gain and logarithmic rank discounting, normalized by ideal DCG (IDCG@K):
   $$\text{DCG}@K = \sum_{r=1}^{K} \frac{2^{\text{rel}_r} - 1}{\log_2(r + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

---

## 4. Empirical Performance Analysis by Category

### 4.1 Granular Performance Across Query Categories

| Category | Query Count | P@1 | R@1 | MRR@1 | NDCG@1 | P@3 | R@3 | MRR@3 | NDCG@3 | P@5 | R@5 | MRR@5 | NDCG@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`amount`** | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | 0.2000 | 1.0000 | 1.0000 | 1.0000 |
| **`date`** | 1 | 1.0000 | 0.5000 | 1.0000 | 1.0000 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | 0.4000 | 1.0000 | 1.0000 | 1.0000 |
| **`exact_entity`** | 3 | 1.0000 | 0.5000 | 1.0000 | 1.0000 | 0.5556 | 0.8333 | 1.0000 | 0.9724 | 0.3333 | 0.8333 | 1.0000 | 0.9724 |
| **`identifier`** | 3 | 1.0000 | 0.5000 | 1.0000 | 1.0000 | 0.3333 | 0.5000 | 1.0000 | 0.9173 | 0.2000 | 0.5000 | 1.0000 | 0.9173 |
| **`semantic_only`** | 6 | 0.8333 | 0.5278 | 0.8333 | 0.8333 | 0.5000 | 0.8333 | 0.9167 | 0.9030 | 0.4000 | 0.9722 | 0.9167 | 0.9272 |
| **`structured_hybrid`** | 4 | 1.0000 | 0.7500 | 1.0000 | 1.0000 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 0.3000 | 1.0000 | 1.0000 | 1.0000 |
| **`no_match`** | 1 | — | — | — | — | — | — | — | — | — | — | — | — |

### 4.2 Structured, Exact-Match, and Hybrid Query Dynamics

The benchmark empirically demonstrates how the hybrid architecture partitions retrieval:
- **Structured Queries (`amount`, `date`, `structured_hybrid`)**: Filtering on extracted metadata attributes (e.g. `amount > 5000`, `date == 2026-01-25`, `type == purchase_order`) eliminates non-matching candidate documents prior to ranking. Consequently, Precision@1, MRR, and NDCG achieve **1.0000** on all structured categories.
- **Exact Entity & Identifier Queries (`exact_entity`, `identifier`)**: The 5-tier lexical match ensures that documents containing matching invoice numbers or supplier names are prioritized (Tier 1/2) over purely semantic matches (Tier 5), achieving **1.0000 MRR**.
- **Semantic-Only Queries (`semantic_only`)**: Without structured constraints or verbatim token overlap, the dense MiniLM embeddings achieve **0.8333 P@1 / MRR@1**, rising to **0.9722 Recall@5** and **0.9272 NDCG@5**.

> **Attribution Notice**: Success on structured and exact-entity queries is attributable to deterministic query parsing and metadata filtering, not to dense semantic vector similarity alone. Aggregate metrics benefit substantially from this hybrid design.

---

## 5. No-Match Query Analysis

- **Query Evaluated**: `"purchase orders over 999999"` (Category: `no_match`)
- **Expected Outcome**: Empty result set (0 documents returned)
- **Observed Outcome**: 0 documents returned (**PASS**)
- **Mechanism**: The query parser extracted a structured constraint `type == purchase_order` AND `amount > 999999`. Because no corpus document satisfied both constraints, pre-filtering returned an empty candidate list before semantic vector search was invoked.

> **Note on Generalization**: While the system correctly returned no false-positive hits for this query (1/1, 100%), this single test query does not constitute an exhaustive evaluation of out-of-domain or adversarial rejection. It validates that the structured filtering pipeline correctly rejects out-of-range numerical filters.

---

## 6. Complete Per-Query Evaluation Trace

| ID | Query String | Category | Gold Relevant Docs | Top Returned Doc IDs | P@1 | R@1 | NDCG@1 | P@5 | R@5 | NDCG@5 | RR |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Q01 | `SuperStore invoice balance due` | `structured_hybrid` | invoice_superstore_39519:3, invoice_superstore_6817:3 | invoice_superstore_6817, invoice_superstore_39519 | 1.00 | 0.50 | 1.00 | 0.40 | 1.00 | 1.00 | 1.00 |
| Q02 | `invoice 39519 Aaron Bergman` | `identifier` | invoice_superstore_39519:3, invoice_superstore_6817:1 | invoice_superstore_39519 | 1.00 | 0.50 | 1.00 | 0.20 | 0.50 | 0.92 | 1.00 |
| Q03 | `invoice 6817 Aaron Hawkins` | `identifier` | invoice_superstore_6817:3, invoice_superstore_39519:1 | invoice_superstore_6817 | 1.00 | 0.50 | 1.00 | 0.20 | 0.50 | 0.92 | 1.00 |
| Q04 | `receipt Quantum Logic Solutions total 13500` | `structured_hybrid` | receipt_quantum_logic_100:3 | receipt_quantum_logic_100 | 1.00 | 1.00 | 1.00 | 0.20 | 1.00 | 1.00 | 1.00 |
| Q05 | `AEON receipt paid by cash GST` | `exact_entity` | receipt_aeon_20180314:3, receipt_quantum_logic_100:1 | receipt_aeon_20180314 | 1.00 | 0.50 | 1.00 | 0.20 | 0.50 | 0.92 | 1.00 |
| Q06 | `purchase order Screenline supplier` | `exact_entity` | po_screenline_5380034300:3, po_screenline_5380034370:3 | po_screenline_5380034370, po_screenline_5380034300 | 1.00 | 0.50 | 1.00 | 0.40 | 1.00 | 1.00 | 1.00 |
| Q07 | `PO number 5380034300` | `identifier` | po_screenline_5380034300:3, po_screenline_5380034370:1 | po_screenline_5380034300 | 1.00 | 0.50 | 1.00 | 0.20 | 0.50 | 0.92 | 1.00 |
| Q08 | `documents from Screenline` | `exact_entity` | po_screenline_5380034300:3, po_screenline_5380034370:3 | po_screenline_5380034370, po_screenline_5380034300, receipt_aeon_20180314, invoice_superstore_39519, invoice_superstore_6817 | 1.00 | 0.50 | 1.00 | 0.40 | 1.00 | 1.00 | 1.00 |
| Q09 | `receipt paid by cash` | `structured_hybrid` | receipt_aeon_20180314:3, receipt_quantum_logic_100:2 | receipt_aeon_20180314, receipt_quantum_logic_100 | 1.00 | 0.50 | 1.00 | 0.40 | 1.00 | 1.00 | 1.00 |
| Q10 | `business document with total amount` | `semantic_only` | invoice_superstore_39519:2, invoice_superstore_6817:2, receipt_quantum_logic_100:2, receipt_aeon_20180314:1, po_screenline_5380034300:1, po_screenline_5380034370:1 | invoice_superstore_39519, invoice_superstore_6817, receipt_quantum_logic_100, receipt_aeon_20180314, po_screenline_5380034300 | 1.00 | 0.17 | 1.00 | 1.00 | 0.83 | 1.00 | 1.00 |
| Q11 | `customer completed payment using cash` | `semantic_only` | receipt_quantum_logic_100:3, receipt_aeon_20180314:3 | receipt_aeon_20180314, receipt_quantum_logic_100, invoice_superstore_39519, invoice_superstore_6817, po_screenline_5380034300 | 1.00 | 0.50 | 1.00 | 0.40 | 1.00 | 1.00 | 1.00 |
| Q12 | `office supplies bought for work` | `semantic_only` | invoice_superstore_39519:3 | invoice_superstore_39519, po_screenline_5380034300, invoice_superstore_6817, po_screenline_5380034370, receipt_aeon_20180314 | 1.00 | 1.00 | 1.00 | 0.20 | 1.00 | 1.00 | 1.00 |
| Q13 | `goods requested for delivery` | `semantic_only` | po_screenline_5380034300:3, po_screenline_5380034370:2 | po_screenline_5380034300, invoice_superstore_6817, invoice_superstore_39519, po_screenline_5380034370, receipt_aeon_20180314 | 1.00 | 0.50 | 1.00 | 0.40 | 1.00 | 0.93 | 1.00 |
| Q14 | `tax included in customer purchase` | `semantic_only` | receipt_aeon_20180314:3 | receipt_aeon_20180314, po_screenline_5380034300, po_screenline_5380034370, invoice_superstore_39519, receipt_quantum_logic_100 | 1.00 | 1.00 | 1.00 | 0.20 | 1.00 | 1.00 | 1.00 |
| Q15 | `items shipped to a customer` | `semantic_only` | invoice_superstore_6817:3 | invoice_superstore_39519, invoice_superstore_6817, po_screenline_5380034300, po_screenline_5380034370, receipt_quantum_logic_100 | 0.00 | 0.00 | 0.00 | 0.20 | 1.00 | 0.63 | 0.50 |
| Q16 | `receipts over 10000` | `amount` | receipt_quantum_logic_100:3 | receipt_quantum_logic_100 | 1.00 | 1.00 | 1.00 | 0.20 | 1.00 | 1.00 | 1.00 |
| Q17 | `purchase orders on 25 January 2026` | `date` | po_screenline_5380034300:3, po_screenline_5380034370:3 | po_screenline_5380034300, po_screenline_5380034370 | 1.00 | 0.50 | 1.00 | 0.40 | 1.00 | 1.00 | 1.00 |
| Q18 | `Screenline purchase orders over 5000` | `structured_hybrid` | po_screenline_5380034300:3 | po_screenline_5380034300 | 1.00 | 1.00 | 1.00 | 0.20 | 1.00 | 1.00 | 1.00 |
| Q19 | `purchase orders over 999999` | `no_match` | *none (no-match)* | *none* | — | — | — | — | — | — | — |

---

## 7. Error & Weak Query Diagnostic Analysis

### 7.1 Analysis of Weak-Performing Semantic Queries

1. **Q15 (`items shipped to a customer`)**:
   - **Gold Document**: `invoice_superstore_6817` (relevance grade 3)
   - **Observed Ranking**: Rank 1: `invoice_superstore_39519` (score 0.3204), Rank 2: `invoice_superstore_6817` (score 0.3014)
   - **Diagnostic Root Cause**: Both documents are SuperStore invoices sharing nearly identical structural vocabulary. The dense embedding for `items shipped to a customer` exhibited slightly higher cosine similarity to invoice 39519 than invoice 6817. Consequently, Precision@1 was 0.0, but the relevant document appeared at rank 2 (MRR@3 = 0.5000, Recall@3 = 1.0000, NDCG@5 = 0.6309).

### 7.2 Precision vs. Recall Dynamics at Increasing K

Across the benchmark, Precision@K declines from **0.9444** at K=1 to **0.3222** at K=5, while Recall@K increases from **0.5926** to **0.8796**:
- **Mathematical Driver**: The 6-document corpus contains an average of only 1.5 relevant documents per query. When retrieving $K=5$ documents, the denominator of Precision@K is fixed at 5, meaning the maximum possible precision for a 1-document query is $\frac{1}{5} = 0.2000$.
- **Ranking Quality**: The sustained high NDCG@5 (**0.9574**) and MRR@5 (**0.9722**) prove that relevant documents are placed at the very top of the list, and the lower Precision@5 is a natural artifact of small ground-truth set sizes.

---

## 8. Methodological Limitations & Dissertation-Safe Scope

When citing these search benchmark results in academic or dissertation contexts, the following methodological boundaries must be explicitly noted:

1. **Corpus & Benchmark Scale**: The evaluation is conducted over a closed corpus of 6 representative business documents and 19 curated queries. It establishes algorithmic correctness and ranking efficacy for the defined pipeline, but does not measure scalability to millions of documents.
2. **Ground Truth Definition**: Document relevance labels and graded scores (grades 1 to 3) were manually assigned based on task domain logic. Relevance judgments in industrial settings may vary across human annotators.
3. **Heterogeneous Query Mix**: The 19 queries deliberately span multiple functional modalities (semantic-only, exact identifier, structured metadata, and hybrid). Performance figures should be cited in terms of this composite mix rather than as pure vector search.
4. **Hybrid Retrieval Attribution**: High aggregate retrieval scores (MRR 0.9722, NDCG 0.9574) reflect the combined strength of deterministic structured parsing, lexical priority tiers, and dense semantic embeddings. Dense embeddings alone should only be credited for the `semantic_only` subset (MRR 0.9167, NDCG 0.9272 at K=5).
5. **Domain Scope**: Results demonstrate efficacy on standard business document types (invoices, receipts, purchase orders) and should not be generalized to open-domain web search or unstructured literary texts.

---

## 9. Conclusion & Final Verification Verdict

The frozen 19-query benchmark rigorously confirms that the IDP search subsystem delivers strong retrieval performance on the defined 19-query benchmark across both structured and unstructured query formulations:
- First-result ranking quality is exceptionally high (Overall MRR = **0.9444** at K=1, rising to **0.9722** at K$\ge$3).
- Dense semantic embeddings retrieved relevant documents effectively for the six semantic-only queries in this benchmark (Semantic-Only NDCG@5 = **0.9272**, Recall@5 = **0.9722**).
- The tested amount and date queries each returned the expected top-ranked result, while the single no-match structured boundary query correctly returned no eligible result (**1/1 PASS**).

**Final Verification Status: VERIFIED & FROZEN**
