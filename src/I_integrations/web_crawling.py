"""
Unified web crawling interface combining DuckDuckGo and Wikipedia functionality.
All functionality is self-contained without external wrapper dependencies.
"""
import os
import sys
import logging
import requests
from typing import Dict, List, Optional, Union, Any, Literal
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required libraries - will be installed via requirements.txt
try:
    from duckduckgo_search import DDGS
    import wikipedia
except ImportError as e:
    logger.error(f"Required library not found: {e}")
    logger.error("Make sure to install required packages: pip install duckduckgo_search wikipedia")
    raise

class WebCrawler:
    """
    Unified web crawler combining DuckDuckGo and Wikipedia search capabilities.
    All functionality is implemented directly in this class (no external wrappers).
    """
    
    def __init__(self):
        """Initialize DuckDuckGo client and Wikipedia."""
        # DuckDuckGo setup
        try:
            self.ddg_client = DDGS()
            logger.info("DuckDuckGo search initialized successfully")
            self.ddg_available = True
        except Exception as e:
            logger.warning(f"Failed to initialize DuckDuckGo search: {e}")
            self.ddg_client = None
            self.ddg_available = False
        
        # Wikipedia setup
        try:
            # Set Wikipedia language to English
            wikipedia.set_lang("en")
            logger.info("Wikipedia API initialized successfully")
            self.wiki_available = True
        except Exception as e:
            logger.warning(f"Failed to initialize Wikipedia: {e}")
            self.wiki_available = False
            
        if not self.ddg_available and not self.wiki_available:
            raise ValueError("Failed to initialize any search APIs")
    
    def search_ddg(
        self, 
        query: str, 
        num_results: int = 5,
        safe_search: str = "moderate",
        region: str = "wt-wt",
        time_limit: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Search the web using DuckDuckGo.
        
        Args:
            query: Search query string
            num_results: Maximum number of results to return
            safe_search: Safety level - 'on', 'moderate', or 'off'
            region: Region code for results
            time_limit: Time restriction (d=day, w=week, m=month, y=year)
            
        Returns:
            List of search results with title, link, and snippet
        """
        if not self.ddg_available:
            logger.warning("DuckDuckGo search unavailable")
            return [{"error": "DuckDuckGo API not available"}]
        
        try:
            logger.info(f"Searching DuckDuckGo for: '{query}'")
            
            # Use DDGS to search
            results = list(self.ddg_client.text(
                query,
                region=region,
                safesearch=safe_search,
                timelimit=time_limit,
                max_results=num_results
            ))
            
            # Format results
            formatted_results = [
                {
                    "title": r.get("title", ""),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", ""),
                    "source": self.extract_domain(r.get("href", "")),
                }
                for r in results if r
            ]
            
            logger.info(f"Found {len(formatted_results)} results from DuckDuckGo")
            return formatted_results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return [{"error": f"Search failed: {str(e)}"}]
    
    def search_wiki(
        self, 
        query: str, 
        num_results: int = 5,
        get_content: bool = False,
        sentences: int = 5
    ) -> Dict[str, Any]:
        """
        Search Wikipedia and retrieve article information.
        
        Args:
            query: Search query string
            num_results: Number of search results to return
            get_content: Whether to retrieve full article content
            sentences: Number of sentences for summary (if not getting full content)
            
        Returns:
            Dictionary with article information and search results
        """
        if not self.wiki_available:
            logger.warning("Wikipedia search unavailable")
            return {"error": "Wikipedia API not available"}
        
        try:
            logger.info(f"Searching Wikipedia for: '{query}'")
            
            # Search for articles
            search_results = wikipedia.search(query, results=num_results)
            
            if not search_results:
                logger.warning(f"No Wikipedia articles found for '{query}'")
                return {"error": "No Wikipedia articles found", "search_query": query}
            
            # Get the most relevant article
            main_article = search_results[0]
            logger.info(f"Found Wikipedia article: {main_article}")
            
            # Get article information based on get_content flag
            try:
                if get_content:
                    # Get full page content
                    page = wikipedia.page(main_article, auto_suggest=True)
                    result = {
                        "title": page.title,
                        "url": page.url,
                        "summary": page.summary,
                        "content": page.content,
                        "references": page.references[:5] if page.references else [],  # Limit references
                        "categories": page.categories[:5] if page.categories else [],  # Limit categories
                        "search_results": search_results
                    }
                else:
                    # Get just the summary
                    summary = wikipedia.summary(main_article, sentences=sentences, auto_suggest=True)
                    result = {
                        "title": main_article,
                        "summary": summary,
                        "search_results": search_results
                    }
                
                return result
                
            except wikipedia.exceptions.DisambiguationError as e:
                # Handle disambiguation by taking the first option
                if e.options:
                    logger.info(f"Disambiguation found. Using first option: {e.options[0]}")
                    if get_content:
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                        result = {
                            "title": page.title,
                            "url": page.url,
                            "summary": page.summary,
                            "content": page.content,
                            "references": page.references[:5] if page.references else [],
                            "categories": page.categories[:5] if page.categories else [],
                            "search_results": search_results
                        }
                    else:
                        summary = wikipedia.summary(e.options[0], sentences=sentences, auto_suggest=False)
                        result = {
                            "title": e.options[0],
                            "summary": summary,
                            "search_results": search_results
                        }
                    return result
                else:
                    return {"error": "Wikipedia disambiguation issue with no options", "search_query": query}
            
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
            return {"error": f"Wikipedia search failed: {str(e)}"}
    
    def search_web(
        self, 
        query: str,
        sources: Union[Literal["all"], Literal["ddg"], Literal["wiki"], List[str]] = "all",
        num_results: int = 5,
        include_wiki_content: bool = False,
        max_wiki_sentences: int = 5,
        safe_search: str = "moderate"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive web research using selected sources.
        
        Args:
            query: Research query string
            sources: Sources to use - "all", "ddg", "wiki", or a list of source names
            num_results: Maximum results per source
            include_wiki_content: Whether to include full Wikipedia article content
            max_wiki_sentences: Maximum sentences in Wikipedia summary
            safe_search: Safety level for DuckDuckGo search
            
        Returns:
            Dictionary with results from all requested sources
        """
        results = {"query": query, "sources_used": []}
        
        # Determine which sources to use
        use_ddg = sources == "all" or sources == "ddg" or (isinstance(sources, list) and "ddg" in sources)
        use_wiki = sources == "all" or sources == "wiki" or (isinstance(sources, list) and "wiki" in sources)
        
        # Add results from each source
        if use_ddg and self.ddg_available:
            ddg_results = self.search_ddg(
                query=query,
                num_results=num_results,
                safe_search=safe_search
            )
            if not isinstance(ddg_results, list) or "error" not in ddg_results[0]:
                results["ddg_results"] = ddg_results
                results["sources_used"].append("ddg")
        
        if use_wiki and self.wiki_available:
            wiki_results = self.search_wiki(
                query=query,
                num_results=num_results,
                get_content=include_wiki_content,
                sentences=max_wiki_sentences
            )
            if "error" not in wiki_results:
                results["wiki_results"] = wiki_results
                results["sources_used"].append("wiki")
        
        # Create a merged context that combines both sources
        context = []
        
        # Add Wikipedia context if available
        if "wiki_results" in results:
            wiki = results["wiki_results"]
            context.append(f"WIKIPEDIA: {wiki.get('title', '')}")
            
            if include_wiki_content and "content" in wiki:
                # Truncate if too long
                content = wiki.get("content", "")
                if len(content) > 2000:
                    content = content[:2000] + "..."
                context.append(content)
            else:
                context.append(wiki.get("summary", ""))
        
        # Add DuckDuckGo context if available
        if "ddg_results" in results and results["ddg_results"]:
            context.append("\nWEB SEARCH RESULTS:")
            
            for i, result in enumerate(results["ddg_results"][:num_results], 1):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                link = result.get("link", "")
                
                context.append(f"{i}. {title}")
                context.append(f"   {snippet}")
                context.append(f"   URL: {link}")
                context.append("")
        
        # Add the merged context to results
        results["merged_context"] = "\n".join(context)
        
        return results
    
    def extract_domain(self, url: str) -> str:
        """Extract the base domain from a URL."""
        import re
        pattern = r'https?://(?:www\.)?([^/]+)'
        match = re.search(pattern, url)
        return match.group(1) if match else url


# Example usage
if __name__ == "__main__":
    print("\n===== WEB CRAWLER DEMO =====\n")
    
    # Initialize the WebCrawler
    try:
        crawler = WebCrawler()
        
        # Get query from user input or use default
        query = input("Enter search query (or press Enter for default): ") or "Python programming language"
        print(f"\nSearching for: '{query}'")
        
        # Test DuckDuckGo search
        if crawler.ddg_available:
            print("\n--- DuckDuckGo Search Results ---")
            ddg_results = crawler.search_ddg(query)
            
            if ddg_results and "error" not in ddg_results[0]:
                for i, result in enumerate(ddg_results[:3], 1):
                    print(f"\n{i}. {result.get('title', '')}")
                    print(f"   {result.get('snippet', '')[:150]}...")
                    print(f"   URL: {result.get('link', '')}")
            else:
                print("DuckDuckGo search failed or returned no results.")
        
        # Test Wikipedia search
        if crawler.wiki_available:
            print("\n--- Wikipedia Search Results ---")
            wiki_result = crawler.search_wiki(query)
            
            if "error" not in wiki_result:
                print(f"Title: {wiki_result.get('title', '')}")
                print(f"Summary: {wiki_result.get('summary', '')[:200]}...")
                if len(wiki_result.get('summary', '')) > 200:
                    print("...")
            else:
                print(f"Wikipedia search failed: {wiki_result.get('error')}")
        
        # Test combined search
        print("\n--- Combined Web Search ---")
        combined = crawler.search_web(query)
        print(f"Sources used: {', '.join(combined.get('sources_used', []))}")
        print("\nMerged Context Preview:")
        merged_context = combined.get('merged_context', '')
        print(merged_context[:500] + "..." if len(merged_context) > 500 else merged_context)
        
    except Exception as e:
        print(f"Error running web crawler demo: {e}")
    
    print("\n===== DEMO COMPLETE =====") 