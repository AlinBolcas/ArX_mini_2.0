"""
OpenWeatherMap API Wrapper
API Docs: https://openweathermap.org/api/one-call-3
Sign up: https://home.openweathermap.org/users/sign_up
Pricing: https://openweathermap.org/price
"""

import os
import requests
from typing import Dict, Optional, List
from dotenv import load_dotenv

load_dotenv()

class OpenWeatherAPI:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenWeather API wrapper."""
        self.api_key = api_key or os.getenv("OPENWEATHERMAP_API_KEY")
        self.base_url = "https://api.openweathermap.org/data"
        
        # Debug logging for initialization
        print("\nOpenWeatherAPI Initialization:")
        # print(f"API Key from env: {os.getenv('OPENWEATHERMAP_API_KEY')}")
        # print(f"Final API Key: {self.api_key}")
        print(f"Base URL: {self.base_url}\n")
        
    def get_current_weather(self, location: str, units: str = "metric") -> Dict:
        """Get current weather for a location."""
        url = f"{self.base_url}/2.5/weather"
        params = {
            "q": location,
            "appid": self.api_key,
            "units": units
        }
        
        # Construct full URL for debugging
        full_url = f"{url}?{'&'.join(f'{k}={v}' for k,v in params.items())}"
        print("\nRequest Details:")
        print(f"Full URL: {full_url}")
        print(f"API Key being used: {self.api_key}")
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            print(f"Response status code: {response.status_code}")
            print(f"Response headers: {response.headers}")
        response.raise_for_status()
        return response.json()
    
    def get_forecast(self, location: str, units: str = "metric", days: int = 5) -> Dict:
        """Get weather forecast for a location."""
        url = f"{self.base_url}/2.5/forecast"
        params = {
            "q": location,
            "appid": self.api_key,
            "units": units,
            "cnt": days * 8  # API returns data in 3-hour steps
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json() 