from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

from .base import BaseLoader
from .documents import DocumentLoader
from .pdf import PDFLoader
from .web import WebLoader
from ..core.models import Document, DocumentType

from ..core.exceptions import DocumentLoadingError, UnsupportedFormatError 
from ..core.logging import get_logger, LogContext


class UnifiedLoader(BaseLoader):
    """
    Orchestrator for all document loading. It determines the source type
    (file, URL, directory) and routes the task to the appropriate sub-loader.
    """
    
    def __init__(self, decodo_client=None):
        super().__init__()
        self.logger = get_logger("loader.UnifiedLoader")

        # Initialize sub-loaders (share DecodoClient with WebLoader if provided)
        self.document_loader = DocumentLoader()
        self.pdf_loader = PDFLoader()

        if decodo_client:
            # Use shared DecodoClient instance for better resource management
            self.web_loader = WebLoader(use_decodo=True)
            self.web_loader.decodo_client = decodo_client
            self.web_loader.use_decodo = True
            self.logger.info("UnifiedLoader using shared Decodo client")
        else:
            self.web_loader = WebLoader()

        # List of all formats we can handle (excluding DocumentType.URL for file path checks)
        self.supported_formats = [
            f.value for f in DocumentType if f != DocumentType.URL
        ]

    async def load_single(
            self,
            source: Union[str, Path],
            **kwargs
    ) -> Document:
        
        # FIX: Added 'source=str(source)' to LogContext for debugging
        with LogContext("unified_load_single", source=str(source)): 
            try:
                # Determine source type and select appropriate loader
                loader = self._select_loader(source)

                if loader is None:
                    # FIX: Renamed variable to avoid confusion with the source path/URL
                    source_format = Path(str(source)).suffix.lstrip('.') if not str(source).startswith(("http", "https")) else "URL"
                    raise UnsupportedFormatError(
                        message=f"Source format '{source_format}' not supported.",
                        file_format=source_format,
                        supported_formats=self.supported_formats
                    )
                
                # Load using Selected loader
                # FIX: Corrected the load function call 
                document = await loader.load_single(source, **kwargs) 
                # FIX: Corrected logging call to show the loader class name
                self.logger.debug(f"Successfully loaded {source} using {loader.__class__.__name__}")

                return document 
            
            except Exception as e:
                # Raise a unified DocumentLoadingError wrapping the underlying exception
                raise DocumentLoadingError(
                    f"Failed to load document from {source}",
                    file_path=str(source),
                    detail=str(e)
                ) from e
            

    async def load_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        # FIX: Added missing comma
        file_patterns: Optional[List[str]] = None, 
        **kwargs
    ) -> List[Document]: # FIX: Corrected return type hint to List[Document]

        directory = Path(directory)

        if not directory.exists() or not directory.is_dir():
            raise DocumentLoadingError(f"Directory not found: {directory}")
    
        # Find all supported files
        sources = self._find_supported_files(
            directory,
            recursive=recursive,
            patterns=file_patterns
        )
    
        if not sources:
            # FIX: Added context to the warning
            self.logger.warning(f"No supported files found in directory {directory}") 
            return []


        self.logger.info(f"Found {len(sources)} supported files in {directory}") # FIX: Corrected typo 'file'
    
        # Load all files
        # NOTE: load_multiple must be defined in BaseLoader or implemented here. 
        # Assuming BaseLoader provides an implementation or we use asyncio.gather on load_single
        return await self.load_multiple(sources, **kwargs)


    def _select_loader(self, source: Union[str, Path]) -> Optional[BaseLoader]:

        """
        Select the appropriate loader for the given source.

        Args:
            source: Source to load

        Returns:
            Appropriate loader or None if unsupported
        """

        try:
            # Handle URLs
            # FIX: Completed URL check
            if isinstance(source, str) and urlparse(source).scheme in ('http', 'https'):
                return self.web_loader
            
            # Handle file path
            source = Path(source) if isinstance(source,str) else source

            if not source.exists():
                return None
            
            # Get file extension
            suffix = source.suffix.lower().lstrip('.')

            try:
                doc_type = DocumentType(suffix)

            except ValueError:
                return None
            
            # Route to appropriate loader
            if doc_type == DocumentType.PDF:
                # FIX: Corrected property access
                return self.pdf_loader
            # FIX: Added closing parenthesis
            elif doc_type in [DocumentType.TXT, DocumentType.MD, DocumentType.HTML, 
                              DocumentType.JSON, DocumentType.CSV, DocumentType.DOCX]:
                return self.document_loader
            else:
                return None
            
        except Exception as e:
            self.logger.error(f"Error selecting loader for {source}: {e}")
            return None
        

    def _find_supported_files(
            self,
            directory: Path,
            recursive: bool = True,
            patterns: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Walks a directory and yields paths of supported files.
        """
        sources = []
        if recursive:
            # Use rglob for recursive searching
            path_iterator = directory.rglob("*")
        else:
            # Use glob for non-recursive searching
            path_iterator = directory.glob("*")
        
        supported_extensions = [f".{ext}" for ext in self.supported_formats]
        
        for p in path_iterator:
            if p.is_file() and p.suffix.lower() in supported_extensions:
                is_excluded_by_pattern = False
                if patterns:
                    # Basic pattern matching (e.g., check if filename contains pattern)
                    for pattern in patterns:
                        if pattern in p.name:
                            is_excluded_by_pattern = True
                            break
                
                if not is_excluded_by_pattern:
                    sources.append(p)
        
        return sources