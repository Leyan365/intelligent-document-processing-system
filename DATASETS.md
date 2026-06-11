# Datasets

## Data Sources

The current classifier and extraction workflow use a local mix of public and
custom datasets:

- RVL-CDIP invoice images: used as invoice source data after OCR text caching.
- SROIE receipt dataset: used as receipt source data by reading receipt OCR box
  text.
- Real purchase order PDFs: used as purchase order source data after local text
  extraction/OCR caching.
- Custom generated text dataset: unified three-class text dataset assembled from
  the cached invoice, purchase order, and receipt text sources.

Dataset files are local-only and are not committed to git.

## Local Folder Structure

Expected source and generated data folders:

```text
data/
  rvl_cdip/
    train/invoice/
    val/invoice/
  sroie/
    train/box/
    test/box/
  real_po_pdfs/
    train/
    val/
  processed/
    rvl_text_cache/
      train/invoice/
      val/invoice/
  custom_po_text/
    train/
    val/
  custom_text_dataset/
    train/
      invoice/
      purchase_order/
      receipt/
    val/
      invoice/
      purchase_order/
      receipt/
```

## Generated Cache Folders

- `data/processed/rvl_text_cache`: cached OCR text extracted from RVL-CDIP
  invoice images.
- `data/custom_po_text`: cached text extracted from real purchase order PDFs.
- `data/custom_text_dataset`: final train/validation text dataset consumed by
  the classifier training script.

## Train And Validation Counts

Latest reported custom text dataset counts:

| Split/Class | Count |
| --- | ---: |
| `invoice_train` | 160 |
| `invoice_val` | 40 |
| `purchase_order_train` | 18 |
| `purchase_order_val` | 6 |
| `receipt_train` | 626 |
| `receipt_val` | 347 |

## Dataset Builder Commands

Build RVL-CDIP invoice OCR text cache:

```powershell
$env:PYTHONPATH='src'; python training/build_rvl_text_cache.py
```

Build purchase order text cache:

```powershell
$env:PYTHONPATH='src'; python training/build_po_text_cache.py
```

Build the unified custom text dataset:

```powershell
$env:PYTHONPATH='src'; python training/build_custom_text_dataset.py
```

## Dataset Limitations

- Source-domain imbalance is present because each class comes from a different
  source family.
- Receipts far outnumber purchase order samples.
- Purchase order samples are based on a small collection of real PDFs.
- RVL-CDIP invoice samples rely on OCR cache quality and can contain noisy text.
- Validation accuracy can be inflated because invoice, receipt, and purchase
  order classes are separated not only by document type but also by source
  dataset characteristics.

## Git Policy

The `data/` directory is ignored by git. This keeps large raw datasets, OCR
caches, generated text datasets, and local experimental files out of version
control.
