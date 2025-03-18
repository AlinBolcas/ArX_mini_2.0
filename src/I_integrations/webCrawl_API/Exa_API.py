import os
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from dotenv import load_dotenv
from exa_py import Exa
import logging

logger = logging.getLogger(__name__)

class ExaWrapper:
    """
    Enhanced wrapper for Exa's API services.
    Provides neural and keyword search with content retrieval capabilities.
    """
    
    def __init__(self):
        """Initialize Exa client with API key from environment."""
        load_dotenv()
        
        api_key = os.getenv("EXA_API_KEY")
        if not api_key:
            raise ValueError("No Exa API key found. Please set EXA_API_KEY environment variable")
            
        self.client = Exa(api_key=api_key)

    def search(
        self,
        query: str,
        num_results: int = 3,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        use_autoprompt: bool = False,
        search_type: str = "auto",
        category: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced search with all Exa options.
        
        Args:
            query: Search query string
            num_results: Number of results
            include_domains: List of domains to restrict search to
            exclude_domains: List of domains to exclude
            start_published_date: Start date for published content (YYYY-MM-DD)
            end_published_date: End date for published content (YYYY-MM-DD)
            start_crawl_date: Start date for crawled content (YYYY-MM-DD)
            end_crawl_date: End date for crawled content (YYYY-MM-DD)
            use_autoprompt: Whether to use Exa's query enhancement
            search_type: 'neural', 'keyword', or 'auto'
            category: Focus category ('company', 'research paper', 'news', etc.)
            include_text: Strings that must be in results (max 1 string, 5 words)
            exclude_text: Strings that must not be in results (max 1 string, 5 words)
        """
        try:
            return self.client.search(
                query=query,
                num_results=num_results,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
                start_crawl_date=start_crawl_date,
                end_crawl_date=end_crawl_date,
                use_autoprompt=use_autoprompt,
                type=search_type,
                category=category,
                include_text=include_text,
                exclude_text=exclude_text
            )
        except Exception as e:
            logger.error(f"Exa search error: {str(e)}")
            return {"results": []}

    def search_with_contents(
        self,
        query: str,
        num_results: int = 3,
        max_chars: int = 1000,
        include_html: bool = False,
        highlights: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search with full text content and highlights.
        
        Args:
            query: Search query string
            num_results: Number of results
            max_chars: Maximum characters to return per result
            include_html: Whether to include HTML tags in content
            highlights: Whether to include content highlights
            **kwargs: Additional search parameters from search()
        """
        try:
            response = self.client.search_and_contents(
                query,
                text={"max_characters": max_chars, "include_html_tags": include_html},
                highlights=highlights,
                num_results=num_results,
                **kwargs
            )
            return response.results  # Return just the results list
        except Exception as e:
            logger.error(f"Exa content search error: {str(e)}")
            return []

    def find_similar(
        self,
        url: str,
        num_results: int = 3,
        with_contents: bool = False,
        max_chars: Optional[int] = None,
        exclude_source_domain: bool = False
    ) -> Dict[str, Any]:
        """
        Find similar documents to a given URL.
        
        Args:
            url: URL to find similar content for
            num_results: Number of results
            with_contents: Whether to include full text content
            max_chars: Maximum characters when with_contents is True
            exclude_source_domain: Whether to exclude the source domain
        """
        try:
            if with_contents:
                return self.client.find_similar_and_contents(
                    url=url,
                    num_results=num_results,
                    text={"max_characters": max_chars} if max_chars else True,
                    exclude_source_domain=exclude_source_domain
                )
            else:
                return self.client.find_similar(
                    url=url,
                    num_results=num_results,
                    exclude_source_domain=exclude_source_domain
                )
        except Exception as e:
            logger.error(f"Exa similarity search error: {str(e)}")
            return {"results": []}