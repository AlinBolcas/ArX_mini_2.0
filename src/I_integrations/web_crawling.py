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
from datetime import datetime
from dotenv import load_dotenv
from exa_py import Exa

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
            print(f"Exa search error: {str(e)}")
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
            # Convert result objects to dictionaries for consistent access
            results_list = []
            for result in response.results:
                result_dict = {
                    "title": result.title if hasattr(result, "title") else "",
                    "url": result.url if hasattr(result, "url") else "",
                    "text": result.text if hasattr(result, "text") else ""
                }
                # Add any other attributes that might be present
                if hasattr(result, "published_date"):
                    result_dict["published_date"] = result.published_date
                if hasattr(result, "author"):
                    result_dict["author"] = result.author
                if hasattr(result, "highlights"):
                    result_dict["highlights"] = result.highlights
                results_list.append(result_dict)
            return results_list
        except Exception as e:
            print(f"Exa content search error: {str(e)}")
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
                response = self.client.find_similar_and_contents(
                    url=url,
                    num_results=num_results,
                    text={"max_characters": max_chars} if max_chars else True,
                    exclude_source_domain=exclude_source_domain
                )
                # Convert result objects to dictionaries
                results_dict = {"results": []}
                if hasattr(response, "results"):
                    for result in response.results:
                        result_dict = {
                            "title": result.title if hasattr(result, "title") else "",
                            "url": result.url if hasattr(result, "url") else "",
                            "text": result.text if hasattr(result, "text") else ""
                        }
                        if hasattr(result, "published_date"):
                            result_dict["published_date"] = result.published_date
                        results_dict["results"].append(result_dict)
                return results_dict
            else:
                response = self.client.find_similar(
                    url=url,
                    num_results=num_results,
                    exclude_source_domain=exclude_source_domain
                )
                # Convert result objects to dictionaries
                results_dict = {"results": []}
                if hasattr(response, "results"):
                    for result in response.results:
                        result_dict = {
                            "title": result.title if hasattr(result, "title") else "",
                            "url": result.url if hasattr(result, "url") else ""
                        }
                        results_dict["results"].append(result_dict)
                return results_dict
        except Exception as e:
            print(f"Exa similarity search error: {str(e)}")
            return {"results": []}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required libraries - will be installed via requirements.txt
try:
    from duckduckgo_search import DDGS
    import wikipedia
except ImportError as e:
    print(f"Required library not found: {e}")
    print("Make sure to install required packages: pip install duckduckgo_search wikipedia")
    raise

class WebCrawler:
    """
    Unified web crawler combining DuckDuckGo, Wikipedia, and Exa search capabilities.
    All functionality is implemented directly in this class (no external wrappers).
    """
    
    def __init__(self):
        """Initialize DuckDuckGo client, Wikipedia, and Exa wrapper."""
        # DuckDuckGo setup
        try:
            self.ddg_client = DDGS()
            print("DuckDuckGo search initialized successfully")
            self.ddg_available = True
        except Exception as e:
            print(f"Warning: Failed to initialize DuckDuckGo search: {e}")
            self.ddg_client = None
            self.ddg_available = False
        
        # Wikipedia setup
        try:
            # Set Wikipedia language to English
            wikipedia.set_lang("en")
            print("Wikipedia API initialized successfully")
            self.wiki_available = True
        except Exception as e:
            print(f"Warning: Failed to initialize Wikipedia: {e}")
            self.wiki_available = False
            
        # Exa API setup
        try:
            self.exa_wrapper = ExaWrapper()
            print("Exa API initialized successfully")
            self.exa_available = True
        except Exception as e:
            print(f"Warning: Failed to initialize Exa API: {e}")
            self.exa_available = False
            
        if not any([self.ddg_available, self.wiki_available, self.exa_available]):
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
            print("Warning: DuckDuckGo search unavailable")
            return [{"error": "DuckDuckGo API not available"}]
        
        try:
            print(f"Searching DuckDuckGo for: '{query}'")
            
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
            
            print(f"Found {len(formatted_results)} results from DuckDuckGo")
            return formatted_results
            
        except Exception as e:
            print(f"Error: DuckDuckGo search failed: {e}")
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
            print("Warning: Wikipedia search unavailable")
            return {"error": "Wikipedia API not available"}
        
        try:
            print(f"Searching Wikipedia for: '{query}'")
            
            # Search for articles
            search_results = wikipedia.search(query, results=num_results)
            
            if not search_results:
                print(f"Warning: No Wikipedia articles found for '{query}'")
                return {"error": "No Wikipedia articles found", "search_query": query}
            
            # Get the most relevant article
            main_article = search_results[0]
            print(f"Found Wikipedia article: {main_article}")
            
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
                    print(f"Disambiguation found. Using first option: {e.options[0]}")
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
            print(f"Error: Wikipedia search failed: {e}")
            return {"error": f"Wikipedia search failed: {str(e)}"}
    
    def search_web(
        self, 
        query: str,
        sources: Union[Literal["all"], Literal["ddg"], Literal["wiki"], Literal["exa"], List[str]] = "all",
        num_results: int = 5,
        include_wiki_content: bool = False,
        max_wiki_sentences: int = 5,
        safe_search: str = "moderate",
        exa_max_chars: int = 2000
    ) -> Dict[str, Any]:
        """
        Perform comprehensive web research using selected sources.
        
        Args:
            query: Research query string
            sources: Sources to use - "all", "ddg", "wiki", "exa", or a list of source names
            num_results: Maximum results per source
            include_wiki_content: Whether to include full Wikipedia article content
            max_wiki_sentences: Maximum sentences in Wikipedia summary
            safe_search: Safety level for DuckDuckGo search
            exa_max_chars: Maximum characters to return per Exa result
            
        Returns:
            Dictionary with results from all requested sources
        """
        results = {"query": query, "sources_used": []}
        
        # Determine which sources to use
        use_ddg = sources == "all" or sources == "ddg" or (isinstance(sources, list) and "ddg" in sources)
        use_wiki = sources == "all" or sources == "wiki" or (isinstance(sources, list) and "wiki" in sources)
        use_exa = sources == "all" or sources == "exa" or (isinstance(sources, list) and "exa" in sources)
        
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
        
        # Add Exa results if requested
        if use_exa and self.exa_available:
            # Use search_with_contents for Exa results
            exa_results = self.exa_wrapper.search_with_contents(
                query=query,
                num_results=num_results,
                max_chars=exa_max_chars
            )
            
            # Add results if we got any
            if exa_results:
                results["exa_results"] = exa_results
                results["sources_used"].append("exa")
        
        # Create a merged context that combines all sources
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
            context.append("\nDDG SEARCH RESULTS:")
            
            for i, result in enumerate(results["ddg_results"][:num_results], 1):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                link = result.get("link", "")
                
                context.append(f"{i}. {title}")
                context.append(f"   {snippet}")
                context.append(f"   URL: {link}")
                context.append("")
        
        # Add Exa context if available
        if "exa_results" in results and results["exa_results"]:
            context.append("\nEXA AI SEARCH RESULTS:")
            
            for i, result in enumerate(results["exa_results"][:num_results], 1):
                title = result.get("title", "")
                content = result.get("text", "")
                url = result.get("url", "")
                
                context.append(f"{i}. {title}")
                if content:
                    context.append(f"   {content}")
                context.append(f"   URL: {url}")
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
        
        # # Test DuckDuckGo search
        # if crawler.ddg_available:
        #     print("\n--- DuckDuckGo Search Results ---")
        #     ddg_results = crawler.search_ddg(query)
            
        #     if ddg_results and "error" not in ddg_results[0]:
        #         for i, result in enumerate(ddg_results[:3], 1):
        #             print(f"\n{i}. {result.get('title', '')}")
        #             print(f"   {result.get('snippet', '')}")
        #             print(f"   URL: {result.get('link', '')}")
        #     else:
        #         print("DuckDuckGo search failed or returned no results.")
        
        # # Test Wikipedia search
        # if crawler.wiki_available:
        #     print("\n--- Wikipedia Search Results ---")
        #     wiki_result = crawler.search_wiki(query)
            
        #     if "error" not in wiki_result:
        #         print(f"Title: {wiki_result.get('title', '')}")
        #         print(f"Summary: {wiki_result.get('summary', '')}")
        #     else:
        #         print(f"Wikipedia search failed: {wiki_result.get('error')}")
        
        # # Test Exa search
        # if hasattr(crawler, 'exa_available') and crawler.exa_available:
        #     print("\n--- Exa Search Results ---")
            
        #     # Basic Exa search
        #     try:
        #         exa_results = crawler.exa_wrapper.search_with_contents(
        #             query=query, 
        #             num_results=2,
        #             max_chars=500
        #         )
                
        #         if exa_results and isinstance(exa_results, list) and len(exa_results) > 0:
        #             for i, result in enumerate(exa_results, 1):
        #                 print(f"\n{i}. {result['title']}")
        #                 if 'text' in result and result['text']:
        #                     print(f"   {result['text']}")
        #                 else:
        #                     print("   No text available")
        #                 print(f"   URL: {result['url']}")
        #         else:
        #             print("Exa search returned no results.")
        #     except Exception as e:
        #         print(f"Exa search error: {str(e)}")
        
        # Test combined search with all sources
        print("\n--- Combined Web Search (All Sources) ---")
        combined = crawler.search_web(
            query=query,
            sources="all",
            num_results=3
        )
        print(f"Sources used: {', '.join(combined.get('sources_used', []))}")
        print("\nMerged Context Preview:")
        merged_context = combined.get('merged_context', '')
        print(merged_context)
        
    except Exception as e:
        print(f"Error running web crawler demo: {e}")
    
    print("\n===== DEMO COMPLETE =====") 