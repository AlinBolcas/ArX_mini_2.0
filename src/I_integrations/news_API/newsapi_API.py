"""
NewsAPI Wrapper
API Docs: https://newsapi.org/docs
Sign up: https://newsapi.org/register
Pricing: https://newsapi.org/pricing

Endpoints:
- /v2/everything: Search all articles
- /v2/top-headlines: Get breaking news headlines
- /v2/top-headlines/sources: Get news sources
"""

import os
import requests
from typing import Dict, Optional, List, Union
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
import json

load_dotenv()

class NewsAPIError(Exception):
    """Custom exception for NewsAPI errors."""
    pass

class NewsAPI:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize NewsAPI wrapper."""
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        self.base_url = "https://newsapi.org/v2"
        
        # Rate limit tracking
        self.daily_limit = 100
        self.calls_today = 0
        self.last_reset = date.today()
        
        # Available categories for top headlines
        self.categories = [
            "business", "entertainment", "general", "health", 
            "science", "sports", "technology"
        ]
        
        # Available languages
        self.languages = [
            "ar", "de", "en", "es", "fr", "he", "it", "nl", 
            "no", "pt", "ru", "sv", "ud", "zh"
        ]
        
        # Available sort options
        self.sort_options = ["relevancy", "popularity", "publishedAt"]
        
        # Debug logging for initialization
        print("\nNewsAPI Initialization:")
        # print(f"API Key from env: {os.getenv('NEWSAPI_KEY')}")
        print(f"Base URL: {self.base_url}\n")
        
    def _handle_response(self, response: requests.Response) -> Dict:
        """Handle API response and track rate limits."""
        # Reset counter if it's a new day
        today = date.today()
        if today != self.last_reset:
            self.calls_today = 0
            self.last_reset = today
            
        try:
            if response.status_code == 200:
                data = response.json()
                self.calls_today += 1
                
                # NewsAPI specific error handling
                if data.get('status') == 'error':
                    raise NewsAPIError(f"API Error: {data.get('message', 'Unknown error')}")
                    
                return data
            elif response.status_code == 401:
                raise NewsAPIError("Invalid API key")
            elif response.status_code == 429:
                raise NewsAPIError("Daily API call limit reached (100 calls). Please try again tomorrow.")
            elif response.status_code >= 500:
                raise NewsAPIError("NewsAPI server error. Please try again later.")
            else:
                raise NewsAPIError(f"API Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            raise NewsAPIError(f"Network error: {str(e)}")
        except json.JSONDecodeError:
            raise NewsAPIError("Invalid JSON response from API")
            
    def _check_rate_limit(self):
        """Check if we've hit the rate limit."""
        if self.calls_today >= self.daily_limit:
            remaining_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1) - datetime.now()
            raise NewsAPIError(
                f"Daily API call limit reached ({self.daily_limit} calls). "
                f"Resets in {remaining_time.seconds//3600} hours and {(remaining_time.seconds//60)%60} minutes."
            )

    def get_everything(
        self,
        query: str = None,
        search_in: Union[str, List[str]] = None,
        sources: Union[str, List[str]] = None,
        domains: Union[str, List[str]] = None,
        exclude_domains: Union[str, List[str]] = None,
        from_date: Union[str, datetime] = None,
        to_date: Union[str, datetime] = None,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 100,
        page: int = 1
    ) -> Dict:
        """Search all articles.
        
        Args:
            query: Keywords or phrases to search for. Advanced search supported:
                  - "exact phrase" for exact match
                  - +must_have_word
                  - -exclude_word
                  - AND / OR / NOT operators
            search_in: Where to search (title, description, content)
            sources: News sources to restrict search to
            domains: Domains to restrict search to
            exclude_domains: Domains to exclude from search
            from_date: Start date for articles
            to_date: End date for articles
            language: 2-letter ISO-639-1 language code
            sort_by: relevancy, popularity, publishedAt
            page_size: Number of results per page (max 100)
            page: Page number
        """
        try:
            self._check_rate_limit()
            url = f"{self.base_url}/everything"
            
            # Handle datetime objects
            if isinstance(from_date, datetime):
                from_date = from_date.isoformat()
            if isinstance(to_date, datetime):
                to_date = to_date.isoformat()
            
            # Handle list parameters
            if isinstance(search_in, list):
                search_in = ','.join(search_in)
            if isinstance(sources, list):
                sources = ','.join(sources)
            if isinstance(domains, list):
                domains = ','.join(domains)
            if isinstance(exclude_domains, list):
                exclude_domains = ','.join(exclude_domains)
            
            params = {
                "apiKey": self.api_key,
                "pageSize": min(page_size, 100),  # Enforce max limit
                "page": page,
                "language": language,
                "sortBy": sort_by
            }
            
            # Add optional parameters
            if query:
                params["q"] = query
            if search_in:
                params["searchIn"] = search_in
            if sources:
                params["sources"] = sources
            if domains:
                params["domains"] = domains
            if exclude_domains:
                params["excludeDomains"] = exclude_domains
            if from_date:
                params["from"] = from_date
            if to_date:
                params["to"] = to_date
            
            response = requests.get(url, params=params, timeout=10)
            return self._handle_response(response)
        except NewsAPIError:
            raise
        except Exception as e:
            raise NewsAPIError(f"Unexpected error during article search: {str(e)}")
    
    def get_top_headlines(
        self,
        query: str = None,
        country: str = None,
        category: str = None,
        sources: Union[str, List[str]] = None,
        page_size: int = 100,
        page: int = 1
    ) -> Dict:
        """Get breaking news headlines.
        
        Args:
            query: Keywords to search in headlines
            country: 2-letter ISO 3166-1 country code
            category: Category of news (business, entertainment, etc.)
            sources: Specific news sources to get headlines from
            page_size: Number of results per page (max 100)
            page: Page number
        """
        try:
            self._check_rate_limit()
            url = f"{self.base_url}/top-headlines"
            
            # Handle list parameters
            if isinstance(sources, list):
                sources = ','.join(sources)
            
            params = {
                "apiKey": self.api_key,
                "pageSize": min(page_size, 100),
                "page": page
            }
            
            # Add optional parameters
            if query:
                params["q"] = query
            if country and not sources:  # country can't be mixed with sources
                params["country"] = country
            if category and not sources:  # category can't be mixed with sources
                params["category"] = category
            if sources:
                params["sources"] = sources
            
            response = requests.get(url, params=params, timeout=10)
            return self._handle_response(response)
        except NewsAPIError:
            raise
        except Exception as e:
            raise NewsAPIError(f"Unexpected error during top headlines search: {str(e)}")
    
    def get_sources(
        self,
        category: str = None,
        language: str = None,
        country: str = None
    ) -> Dict:
        """Get news sources available.
        
        Args:
            category: Category of news sources
            language: 2-letter ISO-639-1 language code
            country: 2-letter ISO 3166-1 country code
        """
        try:
            self._check_rate_limit()
            url = f"{self.base_url}/top-headlines/sources"
            
            params = {
                "apiKey": self.api_key
            }
            
            # Add optional parameters
            if category:
                params["category"] = category
            if language:
                params["language"] = language
            if country:
                params["country"] = country
            
            response = requests.get(url, params=params, timeout=10)
            return self._handle_response(response)
        except NewsAPIError:
            raise
        except Exception as e:
            raise NewsAPIError(f"Unexpected error during sources search: {str(e)}")
    
    def search_by_source(
        self,
        source_id: str,
        query: str = None,
        page_size: int = 100,
        page: int = 1
    ) -> Dict:
        """Convenience method to search articles from a specific source."""
        return self.get_everything(
            query=query,
            sources=source_id,
            page_size=page_size,
            page=page
        )
    
    def get_category_headlines(
        self,
        category: str,
        country: str = "us",
        page_size: int = 100
    ) -> Dict:
        """Convenience method to get headlines for a specific category."""
        if category not in self.categories:
            raise ValueError(f"Invalid category. Must be one of: {self.categories}")
        
        return self.get_top_headlines(
            category=category,
            country=country,
            page_size=page_size
        ) 