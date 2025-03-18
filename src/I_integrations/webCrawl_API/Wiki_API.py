import wikipedia
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class WikiWrapper:
    """
    Clean wrapper for Wikipedia API without Langchain dependencies.
    Provides search, summary, and content retrieval functionality.
    """
    
    def __init__(self, language: str = "en"):
        """
        Initialize Wikipedia API wrapper.
        
        Args:
            language: Wikipedia language code (e.g., 'en', 'es', 'fr')
        """
        wikipedia.set_lang(language)
        
    def search(
        self,
        query: str,
        num_results: int = 10,
        suggestion: bool = True
    ) -> List[str]:
        """
        Search Wikipedia for articles matching the query.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            suggestion: Whether to get search suggestions
            
        Returns:
            List of article titles
        """
        try:
            results = wikipedia.search(
                query,
                results=num_results,
                suggestion=suggestion
            )
            return results if isinstance(results, list) else results[0]
            
        except Exception as e:
            logger.error(f"Wikipedia search error for '{query}': {str(e)}")
            return []

    def get_summary(
        self,
        title: str,
        sentences: int = 5,
        auto_suggest: bool = True
    ) -> str:
        """Get summary of a Wikipedia article."""
        try:
            # Clean the title and try to get the most relevant result
            title = title.split('(')[0].strip()  # Remove parentheses
            title = title.split(',')[0].strip()  # Remove commas
            return wikipedia.summary(
                title,
                sentences=sentences,
                auto_suggest=True  # Always use auto_suggest for better matches
            )
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation pages by taking first suggestion
            if e.options:
                return self.get_summary(e.options[0], sentences, auto_suggest)
            return ""
        except Exception as e:
            logger.debug(f"Wikipedia summary error for '{title}': {str(e)}")
            return ""

    def get_page(
        self,
        title: str,
        auto_suggest: bool = True
    ) -> Dict[str, Any]:
        """Get full Wikipedia page content and metadata."""
        try:
            # Clean the title same way as get_summary
            title = title.split('(')[0].strip()
            title = title.split(',')[0].strip()
            
            page = wikipedia.page(
                title,
                auto_suggest=True,  # Always use auto_suggest
                preload=True
            )
            
            return {
                "title": page.title,
                "url": page.url,
                "content": page.content,
                "summary": page.summary,
                "references": page.references,
                "categories": page.categories,
                "links": page.links,
                "images": page.images
            }
            
        except wikipedia.exceptions.DisambiguationError as e:
            if e.options:
                return self.get_page(e.options[0], auto_suggest)
            return {}
            
        except Exception as e:
            logger.debug(f"Wikipedia page error for '{title}': {str(e)}")
            return {}

    def get_random(self, num_articles: int = 1) -> List[str]:
        """
        Get random Wikipedia article titles.
        
        Args:
            num_articles: Number of random articles to return
            
        Returns:
            List of random article titles
        """
        try:
            return wikipedia.random(num_articles)
            
        except Exception as e:
            logger.error(f"Wikipedia random article error: {str(e)}")
            return []
