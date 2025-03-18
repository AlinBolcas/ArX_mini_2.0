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
        print("\nGNewsAPI Initialization:")
        # print(f"API Key from env: {os.getenv('GNEWS_API_KEY')}")
        print(f"Base URL: {self.base_url}\n")
    
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