import asyncio
import logging
from pathlib import Path
from typing import Union, Optional

from .base import BaseLoader 
from ..core.models import Document, DocumentType
from ..core.exceptions import DocumentLoadError
from ..core.logging import LogContext

logger = logging.getLogger(__name__)

class PDFLoader(BaseLoader):
    """
    Advanced PDF Loader optimized for LLM training data.
    Preserves document structure via Markdown and supports OCR for scanned files.
    """

    def __init__(self):
        super().__init__()
        self.supported_formats = [DocumentType.PDF]

    async def load_single(
            self, 
            source: Union[str, Path],
            **kwargs
    ) -> Document:
        """
        Load a PDF document with layout-aware extraction.

        Args:
            source: PDF file path
            **kwargs: 
                - use_ocr (bool): Force OCR on all pages. Default False.
                - extract_images (bool): Whether to include image placeholders.
        """
        source = Path(source)

        if not source.exists():
            raise DocumentLoadError(f"File not found: {source}")
        
        with LogContext("load_pdf", file=str(source)):
            try:
                # Run the heavy extraction in a separate thread to keep the loop free
                content = await self._extract_pdf_content(source, **kwargs)

                return self.create_document(
                    title=source.stem,
                    content=content,
                    source=source,
                    doc_type=DocumentType.PDF,
                    extraction_method="PDFLoader.pymupdf4llm",
                )
            
            except Exception as e:
                raise DocumentLoadError(
                    f"Failed to load PDF file: {source}",
                    file_path=str(source),
                    cause=e
                )

    async def _extract_pdf_content(self, path: Path, **kwargs) -> str:
        """Internal worker for content extraction."""
        def _process():
            try:
                import fitz  # PyMuPDF
                import pymupdf4llm  # High-level LLM wrapper
            except ImportError:
                raise DocumentLoadError(
                    "Required packages missing. Install with: pip install pymupdf pymupdf4llm"
                )

            # 1. Primary Attempt: Markdown extraction (best for LLMs)
            # This handles tables, headers, and lists automatically.
            md_content = pymupdf4llm.to_markdown(str(path))
            
            # 2. Heuristic Check: Is the extraction suspiciously empty? (Likely a scan)
            if len(md_content.strip()) < 50:
                logger.info(f"Low text yield for {path.name}. Attempting OCR fallback...")
                return self._ocr_fallback(path)

            return md_content

        return await asyncio.to_thread(_process)

    def _ocr_fallback(self, path: Path) -> str:
        """Fallback method using PyMuPDF's integrated Tesseract support."""
        import fitz
        text_parts = []
        
        try:
            with fitz.open(path) as doc:
                for page in doc:
                    # 'get_textpage_ocr' triggers Tesseract on the page image
                    # requires tesseract-ocr system package installed
                    try:
                        tp = page.get_textpage_ocr(flags=3, language="eng")
                        text = page.get_text(textpage=tp, sort=True)
                        if text.strip():
                            text_parts.append(text)
                    except Exception as ocr_err:
                        logger.warning(f"OCR failed on page {page.number}: {ocr_err}")
                        continue
            
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"Total failure during OCR fallback: {e}")
            return ""