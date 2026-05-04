import asyncio
from abc import ABC, abstractmethod
from typing import List, Union, AsyncGenerator, Dict, Any
from pathlib import Path


from ..core.models import Document, DocumentType
from ..core.exceptions import DocumentLoadError, UnsupportedFormatError
from ..core.logging import get_logger, LogContext


class BaseLoader(ABC):
    """
    Abstract base class for all document loaders.

    Provides common functionality and interface that all loaders must implement.
    """


    def __init__(self):
        self.logger = get_logger(f"loader.{self.__class__.__name__}")
        # FIX: Added missing import for List
        self.supported_formats: List[DocumentType] = []

    @abstractmethod
    async def load_single(
        self, 
        source: Union[str, Path], # FIX: Added type hints for source
        **kwargs
    ) -> Document:
        """
        Load a single document from source.

        Args: 
            source: Source path, URL, or identifier
            **kwargs: Additional loading options

        Returns: 
            Loaded document
        
        Raises:
            DocumentLoadError: If loading fails
            UnsupportedFormatError: If format not supported 
        """
        pass # Every loader MUST know how to load one document


    async def load_multiple(
            self, 
            sources: List[Union[str, Path]], # FIX: Added type hints
            max_workers=4,
            **kwargs
        ) -> list[Document]:
        """
        Load multiple documents concurrently.

        Args:
            sources: List of sources to load
            max_workers: Maximum concurrent workers
            **kwargs: Additional loading options

        Returns:
            List of loaded documents
        """
        # FIX: Corrected LogContext typo, should log len(sources)
        with LogContext("load_multiple", source_count=len(sources)):
            semaphore = asyncio.Semaphore(max_workers)

            async def load_with_semaphore(source):
                async with semaphore:
                    try:
                        return await self.load_single(source, **kwargs) # Passed kwargs
                    except Exception as e:
                        # FIX: Improved error logging to include source and exception
                        self.logger.error(f"Failed to load {source}: {e}", exc_info=True)
                        return None
        
            tasks = [load_with_semaphore(source) for source in sources]
            
            # FIX: Corrected syntax error by using return_exceptions=True
            results = await asyncio.gather(*tasks, return_exceptions=True) 

            # Filter out failed loads and exceptions
            documents = []
            for i, result in enumerate(results):
                if isinstance(result, Document):
                    documents.append(result)
                elif isinstance(result, Exception):
                    # FIX: Improved error logging to include source and exception
                    self.logger.error(f"Error loading {sources[i]} due to exception: {result}", exc_info=True)
                # None results (failed loads) are skipped, which is correct

            self.logger.info(f"successfully loaded {len(documents)} out of {len(sources)} sources.")
            return documents
        

    async def load_stream(
            self,
            sources: list[Union[str, Path]],
            **kwargs
    ) -> AsyncGenerator[Document, None]:
        """
        Load documents as a stream (generator)

        Args:
            sources: List of sources to load
            **kwargs: Additional loading options

        Yields: 
            Document as they are loaded
        """
        for source in sources:
            try:
                document = await self.load_single(source, **kwargs)
                yield document
            except Exception as e:
                self.logger.error(f"Failed to load {source}: {e}", exc_info=True) # Added exception info
                continue

    def supports_format(self, doc_type: DocumentType) -> bool:
        """Check if this loader supports the given format."""
        return doc_type in self.supported_formats
    
    def validate_source(self, source: Union[str, Path]) -> bool:
        """
        Validate if the source can be loaded by this loader.

        Args:
            source: Source to validate

        Returns:
            True if source can be loaded
        """
        try:
            if isinstance(source, str):
                # FIX: Corrected startswith syntax for tuple check
                if source.startswith(('http://', 'https://')): 
                    # URL - check if web loader
                    return DocumentType.URL in self.supported_formats
                else:
                    source = Path(source)

            if isinstance(source, Path):
                # FIX: Corrected 'exits' to 'exists'
                if not source.exists(): 
                    return False
                
                # Check file extension
                suffix = source.suffix.lower().lstrip('.')
                try:
                    doc_type = DocumentType(suffix)
                    # FIX: Corrected typo 'supported_format' to 'supports_format'
                    return self.supports_format(doc_type) 
                except ValueError:
                    return False
                
            return True
        
        except Exception:
            return False
        
    def get_document_type(self, source: Union[str, Path]) -> DocumentType:
        """
        Determine document type from source.

        Args:
            source: Source pdf or URL

        Returns:
            DocumentType

        Raises:
            UnsupportedFormatError: If format cannot be determined
        """
        # FIX: Corrected syntax and logic for string check
        if isinstance(source, str):
            # FIX: Corrected startswith syntax for tuple check
            if source.startswith(('http://', "https://")): 
                return DocumentType.URL
            else:
                source = Path(source)

        # FIX: Logic structure ensured Path check runs if source was converted
        if isinstance(source, Path):
            suffix = source.suffix.lower().lstrip('.')
            try:
                return DocumentType(suffix)
            except ValueError:
                raise UnsupportedFormatError(
                    file_format = suffix,
                    supported_formats = [fmt.value for fmt in self.supported_formats]
                )
        # Fallback for unhandled type/source
        raise UnsupportedFormatError(
            file_format = "Unknown",
            supported_formats = [fmt.value for fmt in self.supported_formats]
        )


    def extract_metadata(self, source: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract metadata from source.

        Args:
            source: Source path or URL

        Returns:
            Metadata dictionary
        """
        metadata = {}
        original_source = source # Keep original source reference

        if isinstance(source, str):
            metadata["source"] = source
            if source.startswith(('http://' , 'https://')):
                metadata["source_type"] = "url"
            else:
                metadata["source_type"] = "file"
                source = Path(source) # Convert to Path for file metadata extraction

        if isinstance(source, Path):
            metadata["source"] = str(source.absolute())
            metadata["source_type"] = "file"
            metadata["filename"] = source.name
            metadata["extension"] = source.suffix

            # FIX: Corrected 'exits' to 'exists'
            if source.exists(): 
                stat = source.stat()
                metadata["size"] = stat.st_size
                metadata["modified_time"] = stat.st_mtime
        
        return metadata
    

    def create_document(
        self, 
        title: str,
        content: str,
        source: Union[str, Path],
        doc_type: DocumentType,
        **kwargs
    ) -> Document:
        """
        Create a Document instance with standard metadata.

        Args:
            title: Document title
            content: Document content
            source: Source path or URL
            doc_type: Document type
            **kwargs: Additional document properties

        Returns:
            Document instance
        """
        metadata = self.extract_metadata(source)
        metadata.update(kwargs.get("metadata", {}))

        return Document(
            title=title,
            content=content,
            source=str(source),
            doc_type=doc_type,
            size=len(content.encode('utf-8')),
            metadata=metadata,
            **{k: v for k, v in kwargs.items() if k != "metadata"}
        )