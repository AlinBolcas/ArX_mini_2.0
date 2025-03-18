"""
Tomorrow.io Weather API Wrapper
API Docs: https://docs.tomorrow.io/reference/weather-forecast
Sign up: https://app.tomorrow.io/signup
Pricing: https://www.tomorrow.io/pricing/
"""

import os
import requests
from typing import Dict, Optional, List, Union
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class TomorrowAPI:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Tomorrow.io API wrapper."""
        self.api_key = api_key or os.getenv("TOMORROW_API_KEY")
        self.base_url = "https://api.tomorrow.io/v4"
        
        # Available data layers
        self.core_layers = [
            "temperature", "temperatureApparent", "dewPoint", 
            "humidity", "windSpeed", "windDirection", "windGust",
            "pressureSurfaceLevel", "precipitationIntensity",
            "precipitationProbability", "precipitationType",
            "rainAccumulation", "snowAccumulation", "iceAccumulation",
            "cloudCover", "cloudBase", "cloudCeiling", "visibility"
        ]
        
    def get_realtime(self, location: Union[str, tuple], units: str = "metric") -> Dict:
        """Get realtime weather data.
        
        Args:
            location: City name or (lat, lon) tuple
            units: Units of measurement ("metric" or "imperial")
        """
        url = f"{self.base_url}/weather/realtime"
        
        # Handle location input
        if isinstance(location, str):
            location_param = location
        else:
            location_param = f"{location[0]},{location[1]}"
            
        params = {
            "apikey": self.api_key,
            "location": location_param,
            "units": units
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_forecast(self, location: Union[str, tuple], timesteps: List[str] = ["1h", "1d"],
                    units: str = "metric", startTime: Optional[datetime] = None,
                    endTime: Optional[datetime] = None) -> Dict:
        """Get weather forecast.
        
        Args:
            location: City name or (lat, lon) tuple
            timesteps: Time intervals ["current", "1h", "1d"]
            units: Units of measurement
            startTime: Start time for forecast (default: now)
            endTime: End time for forecast (default: depends on timestep)
        """
        url = f"{self.base_url}/weather/forecast"
        
        # Handle location input
        if isinstance(location, str):
            location_param = location
        else:
            location_param = f"{location[0]},{location[1]}"
            
        params = {
            "apikey": self.api_key,
            "location": location_param,
            "units": units,
            "timesteps": timesteps
        }
        
        if startTime:
            params["startTime"] = startTime.isoformat()
        if endTime:
            params["endTime"] = endTime.isoformat()
            
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_historical(self, location: Union[str, tuple], timesteps: List[str] = ["1h"],
                      startTime: datetime = None, endTime: datetime = None,
                      units: str = "metric") -> Dict:
        """Get historical weather data.
        
        Args:
            location: City name or (lat, lon) tuple
            timesteps: Time intervals ["1h", "1d"]
            startTime: Start time (default: 24h ago)
            endTime: End time (default: now)
            units: Units of measurement
        """
        url = f"{self.base_url}/weather/history"
        
        # Set default time range if not provided
        if not startTime:
            startTime = datetime.now() - timedelta(days=1)
        if not endTime:
            endTime = datetime.now()
            
        # Handle location input
        if isinstance(location, str):
            location_param = location
        else:
            location_param = f"{location[0]},{location[1]}"
            
        params = {
            "apikey": self.api_key,
            "location": location_param,
            "timesteps": timesteps,
            "startTime": startTime.isoformat(),
            "endTime": endTime.isoformat(),
            "units": units
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_climate_normals(self, location: Union[str, tuple], timesteps: List[str] = ["1d"],
                          startTime: datetime = None, endTime: datetime = None) -> Dict:
        """Get historical climate normals.
        
        Args:
            location: City name or (lat, lon) tuple
            timesteps: Time intervals ["1d", "1m"]
            startTime: Start time
            endTime: End time
        """
        url = f"{self.base_url}/weather/forecast/climate"
        
        # Handle location input
        if isinstance(location, str):
            location_param = location
        else:
            location_param = f"{location[0]},{location[1]}"
            
        params = {
            "apikey": self.api_key,
            "location": location_param,
            "timesteps": timesteps
        }
        
        if startTime:
            params["startTime"] = startTime.isoformat()
        if endTime:
            params["endTime"] = endTime.isoformat()
            
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json() 