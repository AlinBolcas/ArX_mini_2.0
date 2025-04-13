"""
GNews API Wrapper
API Docs: https://gnews.io/docs/v4
Sign up: https://gnews.io/register
Pricing: https://gnews.io/pricing

Features:
- Search articles with advanced query syntax
- Get top headlines by category
- Multiple language support
- Country-specific news
- Rich article metadata
"""

import os
import requests
from typing import Dict, Optional, List, Union
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
import json

load_dotenv()

class GNewsAPIError(Exception):
    """Custom exception for GNews API errors."""
    pass

class GNewsAPI:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize GNews API wrapper."""
        self.api_key = api_key or os.getenv("GNEWS_API_KEY")
        self.base_url = "https://gnews.io/api/v4"
        
        # Rate limit tracking
        self.daily_limit = 100
        self.calls_today = 0
        self.last_reset = date.today()
        
        # Available categories
        self.categories = [
            "general", "world", "nation", "business", 
            "technology", "entertainment", "sports", 
            "science", "health"
        ]
        
        # Available languages with their names
        self.languages = {
            "ar": "Arabic", "zh": "Chinese", "nl": "Dutch",
            "en": "English", "fr": "French", "de": "German",
            "el": "Greek", "he": "Hebrew", "hi": "Hindi",
            "it": "Italian", "ja": "Japanese", "ml": "Malayalam",
            "mr": "Marathi", "no": "Norwegian", "pt": "Portuguese",
            "ro": "Romanian", "ru": "Russian", "es": "Spanish",
            "sv": "Swedish", "ta": "Tamil", "te": "Telugu",
            "uk": "Ukrainian"
        }
        
        # Available countries with their names
        self.countries = {
            "au": "Australia", "br": "Brazil", "ca": "Canada",
            "cn": "China", "eg": "Egypt", "fr": "France",
            "de": "Germany", "gr": "Greece", "hk": "Hong Kong",
            "in": "India", "ie": "Ireland", "il": "Israel",
            "it": "Italy", "jp": "Japan", "nl": "Netherlands",
            "no": "Norway", "pk": "Pakistan", "pe": "Peru",
            "ph": "Philippines", "pt": "Portugal", "ro": "Romania",
            "ru": "Russian Federation", "sg": "Singapore",
            "es": "Spain", "se": "Sweden", "ch": "Switzerland",
            "tw": "Taiwan", "ua": "Ukraine", "gb": "United Kingdom",
            "us": "United States"
        }
        
        # Debug logging
        # print("\nGNewsAPI Initialization:")
        # print(f"API Key from env: {os.getenv('GNEWS_API_KEY')}")
        # print(f"Base URL: {self.base_url}\n")
    
    def _handle_response(self, response: requests.Response) -> Dict:
        """Handle API response and track rate limits."""
        # Reset counter if it's a new day
        today = date.today()
        if today != self.last_reset:
            self.calls_today = 0
            self.last_reset = today
            
        try:
            # Debug response
            print(f"\nGNews API Response Status: {response.status_code}")
            print(f"Response Headers: {response.headers}")
            print(f"Response Text: {response.text[:200]}...")  # First 200 chars
            
            if response.status_code == 200:
                self.calls_today += 1
                return response.json()
            elif response.status_code == 403:
                raise GNewsAPIError(
                    "API key invalid or expired. Please:\n"
                    "1. Verify the key is correct\n"
                    "2. Activate the key at https://gnews.io/dashboard\n"
                    "3. Check your subscription status"
                )
            elif response.status_code == 429:
                raise GNewsAPIError("Daily API call limit reached (100 calls). Please try again tomorrow.")
            elif response.status_code == 500:
                raise GNewsAPIError("GNews API server error. Please try again later.")
            else:
                raise GNewsAPIError(f"API Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            raise GNewsAPIError(f"Network error: {str(e)}")
        except json.JSONDecodeError:
            raise GNewsAPIError("Invalid JSON response from API")
            
    def _check_rate_limit(self):
        """Check if we've hit the rate limit."""
        if self.calls_today >= self.daily_limit:
            remaining_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1) - datetime.now()
            raise GNewsAPIError(
                f"Daily API call limit reached ({self.daily_limit} calls). "
                f"Resets in {remaining_time.seconds//3600} hours and {(remaining_time.seconds//60)%60} minutes."
            )

    def search_news(
        self,
        query: str,
        lang: str = "en",
        country: str = "us",
        max_results: int = 10,
        in_fields: List[str] = ["title", "description"],
        nullable: List[str] = None,
        from_date: Union[str, datetime] = None,
        to_date: Union[str, datetime] = None,
        sort: str = "publishedAt",
        page: int = 1,
        expand: bool = False
    ) -> Dict:
        """Search news articles with advanced query support.
        
        Args:
            query: Search query with support for:
                  - "exact phrase"
                  - AND/OR operators
                  - NOT operator
                  - Grouping with ()
            lang: Language code (e.g., 'en', 'fr')
            country: Country code (e.g., 'us', 'gb')
            max_results: Number of results (1-100)
            in_fields: Where to search ['title', 'description', 'content']
            nullable: Fields that can be null ['description', 'content', 'image']
            from_date: Start date (YYYY-MM-DDThh:mm:ssZ)
            to_date: End date (YYYY-MM-DDThh:mm:ssZ)
            sort: Sort by 'publishedAt' or 'relevance'
            page: Page number (paid feature)
            expand: Get full content (paid feature)
        """
        try:
            self._check_rate_limit()
            url = f"{self.base_url}/search"
            
            # Handle datetime objects
            if isinstance(from_date, datetime):
                from_date = from_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            if isinstance(to_date, datetime):
                to_date = to_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            params = {
                "q": query,
                "lang": lang,
                "country": country,
                "max": max_results,
                "apikey": self.api_key
            }
            
            # Add optional parameters
            if in_fields:
                params["in"] = ",".join(in_fields)
            if nullable:
                params["nullable"] = ",".join(nullable)
            if from_date:
                params["from"] = from_date
            if to_date:
                params["to"] = to_date
            if sort:
                params["sortby"] = sort
            if page > 1:
                params["page"] = page
            if expand:
                params["expand"] = "content"
            
            response = requests.get(url, params=params, timeout=10)
            return self._handle_response(response)
        except GNewsAPIError:
            raise
        except Exception as e:
            raise GNewsAPIError(f"Unexpected error during news search: {str(e)}")
    
    def get_top_headlines(
        self,
        category: str = "general",
        lang: str = "en",
        country: str = "us",
        max_results: int = 10,
        nullable: List[str] = None,
        from_date: Union[str, datetime] = None,
        to_date: Union[str, datetime] = None,
        query: str = None,
        page: int = 1,
        expand: bool = False
    ) -> Dict:
        """Get top headlines by category.
        
        Args:
            category: News category (general, world, business, etc.)
            lang: Language code
            country: Country code
            max_results: Number of results
            nullable: Fields that can be null
            from_date: Start date
            to_date: End date
            query: Optional search within headlines
            page: Page number (paid)
            expand: Get full content (paid)
        """
        if category not in self.categories:
            raise ValueError(f"Invalid category. Must be one of: {self.categories}")
            
        url = f"{self.base_url}/top-headlines"
        
        # Handle datetime objects
        if isinstance(from_date, datetime):
            from_date = from_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        if isinstance(to_date, datetime):
            to_date = to_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            
        params = {
            "category": category,
            "lang": lang,
            "country": country,
            "max": max_results,
            "apikey": self.api_key
        }
        
        # Add optional parameters
        if nullable:
            params["nullable"] = ",".join(nullable)
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if query:
            params["q"] = query
        if page > 1:
            params["page"] = page
        if expand:
            params["expand"] = "content"
            
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def search_by_topic(
        self,
        topic: str,
        lang: str = "en",
        country: str = "us",
        max_results: int = 10
    ) -> Dict:
        """Convenience method to search news by topic."""
        return self.search_news(
            query=f'"{topic}"',  # Exact phrase match
            lang=lang,
            country=country,
            max_results=max_results,
            sort="relevance"
        )
    
    def get_category_news(
        self,
        category: str,
        lang: str = "en",
        country: str = "us",
        max_results: int = 10
    ) -> Dict:
        """Convenience method to get news by category."""
        return self.get_top_headlines(
            category=category,
            lang=lang,
            country=country,
            max_results=max_results
        ) 


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
        # print("\nNewsAPI Initialization:")
        # print(f"API Key from env: {os.getenv('NEWSAPI_KEY')}")
        # print(f"Base URL: {self.base_url}\n")
        
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


# --- News Aggregator Class ---

class News:
    def __init__(self, gnews_api_key: Optional[str] = None, newsapi_key: Optional[str] = None):
        """Initializes the aggregator with API clients."""
        # print("\nNews Initialization:")
        try:
            self.gnews = GNewsAPI(api_key=gnews_api_key)
        except Exception as e:
            # print(f"Error initializing GNewsAPI: {e}")
            self.gnews = None # Set to None if initialization fails

        try:
            self.newsapi = NewsAPI(api_key=newsapi_key)
        except Exception as e:
            # print(f"Error initializing NewsAPI: {e}")
            self.newsapi = None # Set to None if initialization fails

        if self.gnews or self.newsapi:
            # print("News Initialized (at least one API client is active).")
            pass
        else:
             print("Warning: News could not initialize any API clients.")


    def _normalize_gnews_article(self, article: Dict) -> Dict:
        """Normalizes a GNews article to a common format."""
        return {
            'title': article.get('title'),
            'description': article.get('description'),
            'url': article.get('url'),
            'published_at': article.get('publishedAt'), # ISO 8601 format
            'source_name': article.get('source', {}).get('name'),
            'content': article.get('content'), # Provided if 'expand=content' used (paid GNews)
            'image_url': article.get('image'),
            'api_provider': 'GNews'
        }

    def _normalize_newsapi_article(self, article: Dict) -> Dict:
        """Normalizes a NewsAPI article to a common format."""
        return {
            'title': article.get('title'),
            'description': article.get('description'),
            'url': article.get('url'),
            'published_at': article.get('publishedAt'), # ISO 8601 format
            'source_name': article.get('source', {}).get('name'),
            'content': article.get('content'), # Sometimes available
            'image_url': article.get('urlToImage'),
            'api_provider': 'NewsAPI'
        }

    def get_news(
        self,
        query: str,
        max_results: int = 20,
        from_date: Union[str, datetime] = None,
        to_date: Union[str, datetime] = None,
    ) -> List[Dict]:
        """
        Fetches and combines news from GNews and NewsAPI based on a query.
        Always searches in English.

        Args:
            query (str): The search keyword or phrase.
            max_results (int): Approximate total number of articles desired.
                               Results will be split between available APIs. Defaults to 20.
            from_date (Union[str, datetime]): The start date for the search.
                                              Can be a datetime object or ISO format string.
            to_date (Union[str, datetime]): The end date for the search.
                                            Can be a datetime object or ISO format string.

        Returns:
            List[Dict]: A list of combined and normalized news articles,
                        sorted by published date (newest first).
                        Returns an empty list if no APIs are available or no results found.
        """
        if not self.gnews and not self.newsapi:
            print("Error: No API clients available in News.")
            return []

        print(f"\nAggregating news for query: '{query}', max_results: {max_results} (Language: en)")
        combined_articles = []
        errors = []

        # Determine how many APIs are active
        active_apis = sum(1 for api in [self.gnews, self.newsapi] if api is not None)
        if active_apis == 0:
             print("No active APIs to fetch from.")
             return []

        # Calculate results per API
        results_per_api = max(1, max_results // active_apis) # Ensure at least 1 result requested if max_results > 0
        gnews_max = results_per_api if self.gnews else 0
        # Adjust NewsAPI max to get closer to the target, handles odd max_results
        newsapi_max = max_results - gnews_max if self.newsapi and self.gnews else (results_per_api if self.newsapi else 0)

        # --- Fetch from GNews ---
        if self.gnews:
            try:
                print(f"Fetching up to {gnews_max} articles from GNews...")
                gnews_from = from_date.strftime("%Y-%m-%dT%H:%M:%SZ") if isinstance(from_date, datetime) else from_date
                gnews_to = to_date.strftime("%Y-%m-%dT%H:%M:%SZ") if isinstance(to_date, datetime) else to_date

                gnews_results = self.gnews.search_news(
                    query=query,
                    max_results=gnews_max,
                    lang="en",
                    from_date=gnews_from,
                    to_date=gnews_to,
                    sort="publishedAt"
                )
                articles = gnews_results.get('articles', [])
                print(f"Received {len(articles)} articles from GNews.")
                for article in articles:
                    combined_articles.append(self._normalize_gnews_article(article))
            except (GNewsAPIError, Exception) as e:
                error_msg = f"Error fetching from GNews: {e}"
                print(error_msg)
                errors.append(error_msg)

        # --- Fetch from NewsAPI ---
        if self.newsapi:
             try:
                print(f"Fetching up to {newsapi_max} articles from NewsAPI...")
                newsapi_from = from_date.isoformat() if isinstance(from_date, datetime) else from_date
                newsapi_to = to_date.isoformat() if isinstance(to_date, datetime) else to_date

                # NewsAPI max page size is 100
                effective_newsapi_max = min(newsapi_max, 100)

                newsapi_results = self.newsapi.get_everything(
                    query=query,
                    page_size=effective_newsapi_max,
                    language="en",
                    from_date=newsapi_from,
                    to_date=newsapi_to,
                    sort_by="publishedAt"
                )
                articles = newsapi_results.get('articles', [])
                print(f"Received {len(articles)} articles from NewsAPI.")
                for article in articles:
                    combined_articles.append(self._normalize_newsapi_article(article))
             except (NewsAPIError, Exception) as e:
                error_msg = f"Error fetching from NewsAPI: {e}"
                print(error_msg)
                errors.append(error_msg)

        # --- Combine and Sort Results ---
        # Sort combined results by published date (descending)
        # Handle potential None values in 'published_at'
        try:
            combined_articles.sort(
                key=lambda x: datetime.fromisoformat(x['published_at'].replace('Z', '+00:00')) if x.get('published_at') else datetime.min,
                reverse=True
            )
        except Exception as e:
            print(f"Warning: Could not sort combined articles by date. Error: {e}")
            # Fallback sort by provider if date sort fails
            combined_articles.sort(key=lambda x: x.get('api_provider', ''))


        print(f"\nAggregation complete. Total articles retrieved: {len(combined_articles)}. Errors encountered: {len(errors)}")
        if errors:
            print("Errors during fetch:")
            for err in errors:
                print(f"- {err}")

        # Ensure we don't exceed max_results if APIs returned more than requested individually
        return combined_articles[:max_results]


# --- Main Execution Block for Testing ---

if __name__ == "__main__":
    print("\n" + "="*40)
    print("  Running News Test Suite")
    print("="*40)

    # Ensure .env file is loaded to get API keys
    # load_dotenv() should have been called already at the top level
    gnews_key = os.getenv("GNEWS_API_KEY")
    newsapi_key = os.getenv("NEWSAPI_KEY")

    if not gnews_key:
        print("Warning: GNEWS_API_KEY not found in environment variables. GNews API calls will fail.")
    if not newsapi_key:
        print("Warning: NEWSAPI_KEY not found in environment variables. NewsAPI calls will fail.")

    # Initialize the aggregator
    aggregator = News(gnews_api_key=gnews_key, newsapi_key=newsapi_key)

    # Define test parameters
    test_query = "artificial intelligence"
    test_max_results = 10 # Aim for 10 total articles

    # --- Test Case 1: Basic Search ---
    print("\n--- Test Case 1: Basic Search ---")
    print(f"Query: '{test_query}', Max Results: {test_max_results}")
    try:
        news_items = aggregator.get_news(
            query=test_query,
            max_results=test_max_results,
        )
        print(f"\nFound {len(news_items)} articles (sorted by date):")
        # Print titles, sources, provider and date for verification
        if news_items:
             for i, item in enumerate(news_items):
                 pub_date = item.get('published_at', 'N/A')
                 try:
                    # Attempt to format date for readability
                    pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
                 except:
                     pass # Keep original string if formatting fails
                 print(f"{i+1}. [{item.get('api_provider')}] {item.get('title')} ({item.get('source_name')}) - {pub_date}")
        else:
             print("No articles found for this query.")

    except Exception as e:
        print(f"An error occurred during Test Case 1: {e}")
        import traceback
        traceback.print_exc()


    # --- Test Case 2: Search with Date Range ---
    print("\n--- Test Case 2: Search with Date Range ---")
    try:
        # Get news from the last 3 days
        to_date_dt = datetime.now()
        from_date_dt = to_date_dt - timedelta(days=3)
        print(f"Query: '{test_query}', Max Results: {test_max_results}, Date Range: {from_date_dt.strftime('%Y-%m-%d')} to {to_date_dt.strftime('%Y-%m-%d')}")

        news_items_dated = aggregator.get_news(
            query=test_query,
            max_results=test_max_results,
            from_date=from_date_dt,
            to_date=to_date_dt
        )
        print(f"\nFound {len(news_items_dated)} articles within the date range (sorted by date):")
        if news_items_dated:
            for i, item in enumerate(news_items_dated):
                 pub_date = item.get('published_at', 'N/A')
                 try:
                    pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
                 except:
                     pass
                 print(f"{i+1}. [{item.get('api_provider')}] {item.get('title')} ({item.get('source_name')}) - {pub_date}")
        else:
             print("No articles found for this query and date range.")

    except Exception as e:
        print(f"An error occurred during Test Case 2: {e}")
        import traceback
        traceback.print_exc()

    # --- Test Case 3: No Results Expected ---
    print("\n--- Test Case 3: Query with No Expected Results ---")
    no_result_query = "zxyqwerpouiaskjdflmnop"
    print(f"Query: '{no_result_query}', Max Results: {test_max_results}")
    try:
        news_items_none = aggregator.get_news(
            query=no_result_query,
            max_results=test_max_results,
        )
        print(f"\nFound {len(news_items_none)} articles.")
        if len(news_items_none) == 0:
            print("Successfully received 0 articles as expected.")
        else:
            print("Warning: Expected 0 articles but received some.")

    except Exception as e:
        print(f"An error occurred during Test Case 3: {e}")
        import traceback
        traceback.print_exc()


    print("\n" + "="*40)
    print("  News Test Suite Finished")
    print("="*40) 