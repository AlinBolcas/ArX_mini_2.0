import os
import sys
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

# Add project root to PYTHONPATH
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.VIII_utils.file_finder import FileFinder
finder = FileFinder()

# Import base class using FileFinder
GoogleBaseAPI = finder.get_class('google_base_API.py', 'GoogleBaseAPI')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GSearchAPI(GoogleBaseAPI):
    """
    Wrapper for Google Custom Search JSON API
    Docs: https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
    """
    
    def __init__(self):
        super().__init__(scopes=['https://www.googleapis.com/auth/customsearch'])
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")
        
        if not self.api_key or not self.cse_id:
            raise ValueError("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID in .env")
            
        try:
            self.service = build(
                "customsearch", "v1",
                developerKey=self.api_key
            )
            logger.info("Google Custom Search API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Custom Search API: {str(e)}")
            raise

    def search(self, 
               query: str,
               site: Optional[str] = None,
               file_type: Optional[str] = None,
               language: str = "en",
               country: Optional[str] = None,
               safe: str = "off",
               num: int = 10,
               start: int = 1) -> Dict:
        """
        Perform a web search with various filters
        
        Args:
            query: Search query string
            site: Limit search to specific site (e.g., "site:example.com")
            file_type: Filter by file type (e.g., "pdf", "doc", "xls")
            language: Search language (e.g., "en", "es", "fr")
            country: Country code for search (e.g., "us", "uk")
            safe: Safe search setting ("off", "medium", "high")
            num: Number of results (1-10)
            start: Starting index (1-based)
            
        Returns:
            Dict containing search results
        """
        try:
            # Build search query with filters
            if site:
                query = f"site:{site} {query}"
            if file_type:
                query = f"filetype:{file_type} {query}"
            
            # Execute search
            result = self.service.cse().list(
                q=query,
                cx=self.cse_id,
                num=min(num, 10),  # API limit is 10
                start=start,
                lr=f"lang_{language}" if language else None,
                cr=f"country{country.upper()}" if country else None,
                safe=safe
            ).execute()
            
            return result
            
        except HttpError as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    def search_images(self,
                     query: str,
                     size: Optional[str] = None,
                     type: Optional[str] = None,
                     color: Optional[str] = None,
                     safe: str = "off",
                     num: int = 10) -> Dict:
        """
        Search for images with filters
        
        Args:
            query: Search query string
            size: Image size ("huge", "icon", "large", "medium", "small", "xlarge", "xxlarge")
            type: Image type ("clipart", "face", "lineart", "stock", "photo", "animated")
            color: Filter by color ("color", "gray", "mono", "trans")
            safe: Safe search setting ("off", "medium", "high")
            num: Number of results (1-10)
            
        Returns:
            Dict containing image search results
        """
        try:
            # Convert size to uppercase if provided
            if size:
                size = size.upper()
            
            result = self.service.cse().list(
                q=query,
                cx=self.cse_id,
                num=min(num, 10),
                searchType="image",
                imgSize=size,
                imgType=type,
                imgDominantColor=color,
                safe=safe
            ).execute()
            
            return result
            
        except HttpError as e:
            logger.error(f"Image search failed: {str(e)}")
            raise

    def search_news(self,
                   query: str,
                   recent: bool = True,
                   language: str = "en",
                   num: int = 10) -> Dict:
        """
        Search for news articles
        
        Args:
            query: Search query string
            recent: Sort by date if True
            language: Search language
            num: Number of results (1-10)
            
        Returns:
            Dict containing news search results
        """
        try:
            # Add news-specific terms to query
            query = f"{query} when:7d" if recent else query
            
            result = self.service.cse().list(
                q=query,
                cx=self.cse_id,
                num=min(num, 10),
                sort="date" if recent else None,
                lr=f"lang_{language}"
            ).execute()
            
            return result
            
        except HttpError as e:
            logger.error(f"News search failed: {str(e)}")
            raise

    def search_videos(self,
                     query: str,
                     duration: Optional[str] = None,
                     num: int = 10) -> Dict:
        """
        Search for videos
        
        Args:
            query: Search query string
            duration: Video duration ("short", "medium", "long")
            num: Number of results (1-10)
            
        Returns:
            Dict containing video search results
        """
        try:
            # Add video-specific terms
            query = f"{query} videoobject"
            
            result = self.service.cse().list(
                q=query,
                cx=self.cse_id,
                num=min(num, 10),
                dateRestrict=duration
            ).execute()
            
            return result
            
        except HttpError as e:
            logger.error(f"Video search failed: {str(e)}")
            raise

    def get_search_info(self) -> Dict:
        """Get information about the search engine configuration"""
        try:
            result = self.service.cse().list(
                cx=self.cse_id,
                q="test"  # Required parameter
            ).execute()
            
            return {
                "search_engine_id": self.cse_id,
                "queries_per_day": result.get("queries", {}).get("nextPage", [{}])[0].get("count", 0),
                "search_time": result.get("searchTime", 0),
                "total_results": result.get("searchInformation", {}).get("totalResults", 0)
            }
            
        except HttpError as e:
            logger.error(f"Failed to get search info: {str(e)}")
            raise 