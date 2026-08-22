# Final Extraction Benchmark Protocol

This document defines the fixed, source-annotated extraction benchmark in
`evaluation/final_extraction_benchmark.csv`. It contains 59 documents: 25
invoices, 25 receipts, and 9 purchase orders.

## Scope and provenance

- The invoice and purchase-order rows are `FINAL_UNSEEN`: selected source files
  were excluded from app-processed records, semantic-search examples, and known
  development/tuning material before annotation.
- The receipt rows are `EXTRACTION_ONLY_CLASSIFIER_VAL`: they are scanned SROIE
  test receipts and are valid for extraction evaluation only, not an independent
  document-classification test.
- All nine listed POs are retained as independent candidates after source-file
  rehashing. They are native-text PDFs. The PO subset spans six issuers and five
  layout families, but shares the Screenline supplier.
- All invoices use one SuperStore synthetic source/template. They provide
  diversity in customer, date, document number, and amount, but not invoice
  layout/source diversity.
- Supplier, date, and total labels for SROIE rows were transcribed from the
  supplied entity annotations and checked against the rendered receipt. Receipt
  document-number labels were manually read from the rendered receipt images.
- No application output, model prediction, OCR/extractor output, or database
  field was used as a gold label.

## Annotation states

Each annotated field has a status of `PRESENT`, `NOT_PRESENT`, or `AMBIGUOUS`.
Blank expected values accompany `NOT_PRESENT` and `AMBIGUOUS` statuses. An
ambiguous status excludes that field/document from value scoring and prevents a
forced, unsupported label. `annotation_status=MANUAL_SOURCE_VERIFIED` means all
fields in the row were annotated or deliberately marked ambiguous from source
material.

`currency_expected` records the literal currency designation printed by the
source, separately from the decimal amount. A blank currency means no explicit
currency designation was available and currency is not scored.

## Normalization and exact-match rules

- **Amount:** remove thousands separators; retain the exact decimal value to two
  places; do not retain a currency symbol in the amount.
- **Currency:** compare the separately recorded source designation exactly after
  trimming and uppercasing.
- **Date:** represent only unambiguous dates as `YYYY-MM-DD`. Where a numeric
  day/month order cannot be resolved from the document, leave the expected value
  blank and set `date_status=AMBIGUOUS`.
- **Supplier:** trim leading/trailing whitespace, collapse internal whitespace,
  and compare case-insensitively. Do not remove punctuation or expand legal
  suffixes.
- **Document number:** trim and collapse whitespace, compare case-insensitively,
  and retain punctuation such as hyphens and slashes. Do not infer an identifier
  from an unlabelled register, reference, or POS sequence.

## Evaluation definition (not yet run)

The primary benchmark is extraction under the manifest's gold
`document_type`; document classification is not part of its headline score. A
later end-to-end experiment may report type routing separately, with the same
manifest and a clear denominator.

For each field, report:

1. presence/absence precision, recall, and F1 over non-ambiguous status labels;
2. value exact-match accuracy conditional on gold `PRESENT`; and
3. support, excluded-ambiguous count, and `NOT_PRESENT` count.

Report the same measures overall and by document type. Do not run or publish
final extraction metrics until the manifest and normalization rules are
accepted as frozen.

## Freeze procedure

The CSV SHA-256 is the benchmark freeze identifier. Any edit to a selected
source, source hash, gold value, status, normalization rule, or row composition
requires a new manifest version and hash; prior results remain tied to the old
hash.

Frozen manifest SHA-256: `04016ac551e0a5dc8a9136085f6025ce7f880113b2953dbc2154d6fa6301b592`.

## Versioned benchmark freezes

### V1 - historical baseline

- Manifest: `evaluation/final_extraction_benchmark.csv`
- Documents: 59 (25 invoices, 25 receipts, 9 purchase orders)
- SHA-256: `04016ac551e0a5dc8a9136085f6025ce7f880113b2953dbc2154d6fa6301b592`
- Field-status limitation: no `NOT_PRESENT` gold cases.

V1 remains immutable historical evidence and is not replaced or edited by V2.

### V2 - final extraction scoring manifest

- Manifest: `evaluation/final_extraction_benchmark_v2.csv`
- Documents: 63 (29 invoices, 25 receipts, 9 purchase orders)
- SHA-256: `ae7e45d4b273002da476c54208c346cc31f5c53b473adc3900da89bf6a9774d9`
- Addition: four naturally occurring, native-text, zero-total invoice shells
  selected before final system scoring. Each has a visible invoice title, date,
  `$0.00` balance due, and `$0.00` total, while supplier and document number are
  genuinely absent.

V2 adds four `NOT_PRESENT` supplier cases and four `NOT_PRESENT` document-number
cases. It improves presence/absence evaluation for those two fields only; date
and amount still have no `NOT_PRESENT` gold cases. The source-pool invoice shells
are legitimate invoice documents and were selected without using any application,
OCR, extractor, or model output.
