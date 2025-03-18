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

# Example usage
if __name__ == "__main__":
    print("\n===== OPENWEATHER API TEST =====\n")
    
    # Initialize API
    api = OpenWeatherAPI()
    
    # Test locations
    test_locations = [
        "London,UK",
        "New York,US",
        "Tokyo,JP",
        "Sydney,AU"
    ]
    
    # Get current weather for multiple locations
    print("\n=== Current Weather ===")
    for location in test_locations:
        try:
            print(f"\nWeather for {location}:")
            weather = api.get_current_weather(location)
            
            # Extract and display key weather information
            if "main" in weather and "weather" in weather:
                temp = weather["main"].get("temp", "N/A")
                feels_like = weather["main"].get("feels_like", "N/A")
                humidity = weather["main"].get("humidity", "N/A")
                description = weather["weather"][0].get("description", "N/A") if weather["weather"] else "N/A"
                
                print(f"Temperature: {temp}°C")
                print(f"Feels like: {feels_like}°C")
                print(f"Humidity: {humidity}%")
                print(f"Description: {description}")
            else:
                print("Weather data not available")
                
        except Exception as e:
            print(f"Error getting weather for {location}: {e}")
    
    # Get forecast for a location
    print("\n\n=== Weather Forecast ===")
    forecast_location = "Stamford, UK"
    try:
        print(f"\nForecast for {forecast_location}:")
        forecast = api.get_forecast(forecast_location, days=3)
        
        if "list" in forecast:
            # Display the first few forecast entries
            for i, entry in enumerate(forecast["list"][:5]):
                if i == 0:
                    print("\nUpcoming weather:")
                    
                # Extract time and weather information
                dt_txt = entry.get("dt_txt", "N/A")
                temp = entry.get("main", {}).get("temp", "N/A")
                description = entry.get("weather", [{}])[0].get("description", "N/A") if entry.get("weather") else "N/A"
                
                print(f"{dt_txt}: {temp}°C, {description}")
        else:
            print("Forecast data not available")
            
    except Exception as e:
        print(f"Error getting forecast for {forecast_location}: {e}")
        
    print("\n===== TEST COMPLETE =====") 