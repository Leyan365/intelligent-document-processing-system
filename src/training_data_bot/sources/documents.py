import json
import csv 
import asyncio
import docx 
import bs4
from pathlib import Path
from typing import Union, Dict, Any, List, cast 


from .base import BaseLoader 
from ..core.models import Document, DocumentType
from ..core.exceptions import DocumentLoadError
from ..core.logging import LogContext


class DocumentLoader(BaseLoader):
    """
    Loader for text-based document formats.

    Supports TXT, MD, HTML, JSON, CSV, DOCX
    """

    def __init__(self):
        super().__init__()
        self.supported_formats = [
            DocumentType.TXT,
            DocumentType.MD,
            DocumentType.HTML,
            DocumentType.JSON,
            DocumentType.CSV,
            DocumentType.DOCX,
        ]
        

    async def load_single(
            self, 
            source: Union[str, Path],
            encoding: str = "utf-8",
            **kwargs
    ) -> Document:
        """
        Load a single text-based document

        Args:
            source: File path
            encoding: Text encoding
            **kwargs

        Returns:
            Loaded document
        """

        source = Path(source)

        if not source.exists():
            raise DocumentLoadError(f"File not found: {source}")
        
        doc_type = self.get_document_type(source)

        with LogContext("load_document", file=str(source), type_=doc_type.value):
            try:
                # Route to appropriate loader method
                if doc_type == DocumentType.TXT:
                    content = await self._load_text(source, encoding)
                elif doc_type == DocumentType.MD:
                    content = await self._load_markdown(source, encoding)
                elif doc_type == DocumentType.HTML:
                    content = await self._load_html(source, encoding)
                elif doc_type == DocumentType.JSON:
                    content = await self._load_json(source, encoding)
                elif doc_type == DocumentType.CSV:
                    content = await self._load_csv(source, encoding)
                elif doc_type == DocumentType.DOCX:
                    content = await self._load_docx(source)
                else:
                    raise DocumentLoadError(f"Unsupported format: {doc_type}")
                

                title = source.stem
                
                document = self.create_document(
                    title=title,
                    content=content,
                    source=source,
                    doc_type=doc_type,
                    encoding=encoding,
                    extraction_method=f"DocumentLoader.{doc_type.value}",
                    **kwargs 
                )

                return document
            
            except Exception as e:

                raise DocumentLoadError(
                    f"Failed to load {doc_type.value} file: {source}",
                    file_path=str(source),
                    cause=e
                )


    async def _load_text(self, path: Path, encoding: str) -> str:
        """
        Load plain text file. Uses asyncio.to_thread for blocking file I/O.
        """
        return await asyncio.to_thread(path.read_text, encoding=encoding)
    
    async def _load_markdown(self, path: Path, encoding: str) -> str:
        """
        Load Markdown file. Uses asyncio.to_thread for blocking file I/O.
        """
        # For now treat as plain text
        return await asyncio.to_thread(path.read_text, encoding=encoding)
    
    
    # NOTE: This method involves blocking I/O (file open) and CPU work (BeautifulSoup).
    # It must be wrapped in asyncio.to_thread for production use.
    def _extract_text_from_html(self, path: Path, encoding: str) -> str:
        """Helper function for synchronous HTML parsing."""
        from bs4 import BeautifulSoup
        
        with open(path, 'r', encoding=encoding) as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            
        # Decompose scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()
            
        text = soup.get_text()
        
        # Clean up text by removing extra spaces and newlines
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
        
        return ' '.join(chunk for chunk in chunks if chunk)


    async def _load_html(self, path: Path, encoding: str) -> str:
        """
        Load HTML file and extract text content using BeautifulSoup.
        """
        try:
            content = await asyncio.to_thread(self._extract_text_from_html, path, encoding)
            return content
        except ImportError:
            # Fallback if BeautifulSoup is not installed
            return await asyncio.to_thread(path.read_text, encoding=encoding)


    async def _load_json(self, path: Path, encoding: str) -> str:
        """
        Load JSON file and convert content to a readable string format.
        """
        def _parse_json_sync(p: Path, enc: str) -> str:
            with open(p, 'r', encoding=enc) as f:
                data = json.load(f)
            
            lines: List[str] = []
            if isinstance(data, dict):
                lines = [f"{key}: {value}" for key, value in data.items()]
            elif isinstance(data, list):
                lines = [f"Item {i+1}: {item}" for i, item in enumerate(data)]
                
            return "\n".join(lines)
        
        return await asyncio.to_thread(_parse_json_sync, path, encoding)


    async def _load_csv(self, path: Path, encoding: str) -> str:
        """
        Load CSV file and convert content to a structured string format.
        """
        def _parse_csv_sync(p: Path, enc: str) -> str:
            lines: List[str] = []
            with open(p, 'r', encoding=enc, newline='') as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                
                if headers:
                    lines.append("Headers: " + ", ".join(headers))
                    lines.append("")
                    
                for row_num, row in enumerate(reader, 1):
                    # Added check to ensure row length matches headers
                    if headers and len(row) == len(headers):
                        row_data = [f"{header}: {value}" for header, value in zip(headers, row)]
                        lines.append(f"Row {row_num}: {' | '.join(row_data)}")
                
                return "\n".join(lines)
        
        return await asyncio.to_thread(_parse_csv_sync, path, encoding)


    async def _load_docx(self, path: Path) -> str:
        """
        Load DOCX file and extract text content.
        """
        def _parse_docx_sync(p: Path) -> str:
            try:
                from docx import Document
                doc = Document(p)
                text_parts = [p.text for p in doc.paragraphs if p.text.strip()]
                return "\n".join(text_parts)
            except ImportError:
                raise DocumentLoadError("python-docx package required for DOCX files. Please install it with 'pip install python-docx'")
        
        return await asyncio.to_thread(_parse_docx_sync, path)