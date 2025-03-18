from typing import List, Dict, Optional, Any
from duckduckgo_search import DDGS
import logging

logger = logging.getLogger(__name__)

class DDGWrapper:
    """
    Clean wrapper for DuckDuckGo search API without Langchain dependencies.
    Provides text search, news search, and image search functionality.
    """
    
    def __init__(self):
        """Initialize DuckDuckGo search client."""
        self.client = DDGS()
        
    def search(
        self,
        query: str,
        num_results: int = 10,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform a text search using DuckDuckGo.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            region: Region code for search results
            safesearch: 'on', 'moderate', or 'off'
            timelimit: Time limit for results (e.g., 'd', 'w', 'm', 'y')
            
        Returns:
            List of search results with metadata
        """
        try:
            results = list(self.client.text(
                query,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                max_results=num_results
            ))
            
            return [
                {
                    "title": r.get("title", ""),
                    "link": r.get("link", ""),
                    "snippet": r.get("body", ""),
                    "source": r.get("source", ""),
                    "published": r.get("published", "")
                }
                for r in results if r
            ]
            
        except Exception as e:
            print(f"DuckDuckGo search error: {str(e)}")
            return []

    def search_news(
        self,
        query: str,
        num_results: int = 10,
        region: str = "wt-wt",
        timelimit: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles using DuckDuckGo.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            region: Region code for search results
            timelimit: Time limit for results (e.g., 'd', 'w', 'm', 'y')
            
        Returns:
            List of news articles with metadata
        """
        try:
            results = list(self.client.news(
                query,
                region=region,
                timelimit=timelimit,
                max_results=num_results
            ))
            
            return [
                {
                    "title": r.get("title", ""),
                    "link": r.get("link", ""),
                    "snippet": r.get("body", ""),
                    "source": r.get("source", ""),
                    "published": r.get("date", ""),
                    "image": r.get("image", "")
                }
                for r in results if r
            ]
            
        except Exception as e:
            print(f"DuckDuckGo news search error: {str(e)}")
            return []

    def search_images(
        self,
        query: str,
        num_results: int = 10,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        size: Optional[str] = None,
        color: Optional[str] = None,
        type_image: Optional[str] = None,
        layout: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for images using DuckDuckGo.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            region: Region code for search results
            safesearch: 'on', 'moderate', or 'off'
            size: Filter by size ('small', 'medium', 'large')
            color: Filter by color
            type_image: Filter by type ('photo', 'clipart', 'gif', 'transparent')
            layout: Filter by layout ('square', 'tall', 'wide')
            
        Returns:
            List of image results with metadata
        """
        try:
            results = list(self.client.images(
                query,
                region=region,
                safesearch=safesearch,
                size=size,
                color=color,
                type_image=type_image,
                layout=layout,
                max_results=num_results
            ))
            
            return [
                {
                    "title": r.get("title", ""),
                    "image": r.get("image", ""),
                    "thumbnail": r.get("thumbnail", ""),
                    "source": r.get("source", ""),
                    "height": r.get("height", 0),
                    "width": r.get("width", 0)
                }
                for r in results if r and r.get("image")  # Only return results with images
            ]
            
        except Exception as e:
            logger.error(f"DuckDuckGo image search error: {str(e)}")
            return []
