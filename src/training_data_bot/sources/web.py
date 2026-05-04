import asyncio
from typing import Union, Optional, Tuple, List
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from .base import BaseLoader 
from ..core.models import Document, DocumentType
from ..core.exceptions import DocumentLoadingError
from ..core.logging import LogContext, get_logger
from ..decodo import DecodoClient


class WebLoader(BaseLoader):
    """
    Loader for web content from URLs using Decodo's professional scraping service

    Features:
    - Professional web scraping with Decodo API
    - Intelligent fallback to basic scraping
    - Javascript rendering and dynamic content
    - Bypass bot detection and rate limiting
    - Clean text extraction from any website
    """

    # NOTE: The DecodoClient is often injected by UnifiedLoader for shared resource management
    def __init__(self, use_decodo: bool = True, **decodo_kwargs):
        """
        Initialize loader with Decodo integration.

        Args:
            use_decodo: Whether to use Decodo for scraping (defaults: True)
            **decodo_kwargs: Additional arguments for Decodo client 
        """

        super().__init__()
        # FIX: Corrected typo 'wen_loader'
        self.logger = get_logger("web_loader") 

        # State and configuration
        self.supported_format = [DocumentType.URL]
        self.use_decodo = use_decodo
        # FIX: Corrected type hint syntax
        self.decodo_client: Optional[DecodoClient] = None 

        if self.use_decodo:
            try:
                # Initialize Decodo client if not already set (e.g., by UnifiedLoader injection)
                if not self.decodo_client:
                    self.decodo_client = DecodoClient(**decodo_kwargs)
                self.logger.info("WebLoader initialized with Decodo Professional scraping.")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Decodo client: {e}")
                self.logger.info("WebLoader will fallback to basic scraping")
                self.use_decodo = False

    async def load_single(
            self,
            source: Union[str],
            **kwargs
    ) -> Document:
        """
        Load content from a URL using Decodo's professional scraping.
        
        ... [docstring truncated for brevity] ...
        """

        if not isinstance(source, str) or not source.startswith(('http://', 'https://')):
            raise DocumentLoadingError(f"Invalid URL: {source}")
        
        # FIX: Added closing parenthesis
        with LogContext("load url", url=source, method="decodo" if self.use_decodo else "fallback"):
            try:
                # Try Decodo professional scraping first
                if self.use_decodo and self.decodo_client:
                    # FIX: Corrected method name to match convention
                    content, extraction_method = await self._fetch_with_decodo(source, **kwargs) 
                else:
                    content, extraction_method = await self._fetch_with_fallback(source)

                # Extract title from content or URL
                title = self._extract_title(source, content)

                # FIX: Corrected local variable name (was overwriting the Model class)
                document = self.create_document(
                    title=title,
                    content=content,
                    source=source,
                    doc_type=DocumentType.URL,
                    extraction_method=extraction_method,
                )

                # FIX: Completed logging string
                self.logger.info(f"Successfully loaded {len(content)} characters from {source}") 
                return document
            
            except Exception as e:
                self.logger.error(f"Failed to load URL {source}: {e}") # FIX: Corrected typo 'Falied'
                raise DocumentLoadingError(
                    f"Failed to load URL: {source}", # FIX: Corrected typo 'Falied'
                    # FIX: Corrected typo 'sorted' to 'source'
                    file_path=source, 
                    detail=str(e) # Using detail=str(e) for better exception consistency
                ) from e
            
    async def _fetch_with_decodo(self, url: str, **kwargs) -> Tuple[str, str]: # FIX: Added colon and corrected to internal method name
        """
        Fetch content using Decodo's professional scraping service.

        ... [docstring truncated for brevity] ...
        """
        try:
            self.logger.debug(f"Using Decodo professional scraping for {url}")

            # Set up Decodo parameters
            scrape_params = {
                "target": kwargs.get("target", "universal"),
                "locale": kwargs.get("locale", "en-us"),
                "geo": kwargs.get("geo", "United States"),
                "device_type": kwargs.get("device_type", "desktop"),
                "output_format": kwargs.get("output_format", "text") # Changed default to 'text' for cleaner extraction
            }


            # Call Decodo API
            result = await self.decodo_client.scrape_url(url, **scrape_params)

            # Extract content from Decodo response
            if isinstance(result, dict):
                if "content" in result and result["content"]:
                    content = result["content"]
                    if len(content.strip()) > 0:
                        self.logger.debug(f"Decodo extracted {len(content)} characters")
                        return content, "webLoader.Decodo.API" # Changed method name to reflect API use
                    
            # If we get here, Decodo didn't return usable content
            self.logger.warning(f"Decodo returned unusable content for {url}") # FIX: Corrected typo 'unsuable'
            return await self._fetch_with_fallback(url)
        
        except Exception as e:
            self.logger.warning(f"Decodo scraping failed for {url}: {e}")
            self.logger.info("Falling back to basic scraping")
            return await self._fetch_with_fallback(url)

    # --- NEW HELPER METHODS ---

    async def _fetch_with_fallback(self, url: str) -> Tuple[str, str]:
        """
        Basic HTTP fetching using httpx and BeautifulSoup for text extraction.
        Used as a reliable fallback when the Decodo service fails or is disabled.
        """
        self.logger.debug(f"Using fallback basic scraping for {url}")
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
                response = await client.get(url, headers={'User-Agent': 'TrainingDataBot/1.0'})
                response.raise_for_status() # Raise exception for 4xx/5xx status codes
            
            # Use BeautifulSoup to strip HTML and extract clean text
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Prioritize article/main content if available
            main_content = soup.find('article') or soup.find('main')
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                # Fallback to entire body text
                text = soup.body.get_text(separator=' ', strip=True) if soup.body else soup.get_text(separator=' ', strip=True)
                
            return text, "webLoader.Fallback.HTTPX_BS4"
            
        except httpx.HTTPStatusError as e:
            raise DocumentLoadingError(f"HTTP error {e.response.status_code} during fallback scrape: {url}") from e
        except Exception as e:
            raise DocumentLoadingError(f"Basic fallback scraping failed for {url}: {e}") from e

    def _extract_title(self, url: str, content: str) -> str:
        """Extracts title from content (HTML page) or generates one from the URL."""
        
        # 1. Try to extract title from content (assuming it's HTML when scraped, 
        # or if Decodo returned text, it might be the first line)
        try:
            soup = BeautifulSoup(content, 'html.parser')
            title_tag = soup.find('title')
            if title_tag and title_tag.string and title_tag.string.strip():
                return title_tag.string.strip()
        except Exception:
            pass # Not valid HTML, proceed to next fallback

        # 2. Generate title from URL path
        try:
            path = urlparse(url).path.strip('/')
            if path:
                # Format path segments (e.g., 'a/b/c' -> 'C B A')
                segments = [s.replace('-', ' ').title() for s in path.split('/') if s]
                if segments:
                    return ' - '.join(reversed(segments))
        except Exception:
            pass
            
        # 3. Final fallback
        return f"Web Document from {urlparse(url).netloc}"

    # --- MULTI-URL LOADING ---

    async def load_multiple(
            self, # NOTE: Changed method name from load_multiple_urls to load_multiple 
                  # to align with expected BaseLoader interface and make it general
            urls: List[str],
            max_concurrent: int = 5,
            **kwargs
    ) -> List[Document]:
        """
        Load multiple URLs concurrently.

        Args:
            urls: List of URLs to load
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional scraping options

        Returns:
            List of loaded documents
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def load_with_semaphore(url: str) -> Optional[Document]:
            async with semaphore:
                try:
                    return await self.load_single(url, **kwargs)
                except Exception as e:
                    self.logger.error(f"Failed to load {url}: {e}")
                    return None
                
        tasks = [load_with_semaphore(url) for url in urls]
        # NOTE: return_exceptions=False is safer here as we handle exceptions in load_with_semaphore
        results = await asyncio.gather(*tasks) 


        # Filter out None results
        documents = [result for result in results if result is not None]

        self.logger.info(f"Successfully loaded {len(documents)} / {len(urls)} documents")
        return documents


    async def close(self):
        """Clean up resources"""
        if self.decodo_client:
            await self.decodo_client.close()
            self.logger.debug("Decodo client closed")

            
    async def __aenter__(self):
        """Async contextmanager entry."""
        return self
    

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()