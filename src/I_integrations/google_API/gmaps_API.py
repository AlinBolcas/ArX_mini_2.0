"""
Google Maps API Wrapper
Comprehensive wrapper for Google Maps Platform APIs
"""

import os
import sys
from pathlib import Path
import googlemaps
from typing import Dict, Optional, List, Union, Tuple, BinaryIO
from datetime import datetime
from dotenv import load_dotenv
import requests
import re

# Add project root to PYTHONPATH
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.VIII_utils.file_finder import FileFinder
finder = FileFinder()

# Import base class using FileFinder
GoogleBaseAPI = finder.get_class('google_base_API.py', 'GoogleBaseAPI')

load_dotenv()

class GMapsAPIError(Exception):
    """Custom exception for Google Maps API errors."""
    pass

class GMapsAPI(GoogleBaseAPI):
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Google Maps API wrapper."""
        super().__init__(scopes=['https://www.googleapis.com/auth/maps'])
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        self.client = googlemaps.Client(key=self.api_key)
        
        # Available place types
        self.place_types = [
            "restaurant", "cafe", "bar", "lodging", "airport",
            "hospital", "park", "museum", "shopping_mall"
        ]
        
        print("\nGoogle Maps API Initialization:")
        print(f"Client initialized with API key: {'Yes' if self.api_key else 'No'}\n")

        # Add output directory for static maps and street view images
        self.output_dir = Path("output/maps")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Places API Methods
    def text_search(
        self, 
        query: str,
        location: Optional[Tuple[float, float]] = None,
        radius: Optional[int] = None,
        language: str = "en",
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        open_now: bool = False
    ) -> Dict:
        """
        Search for places using a text query.
        Example: "restaurants in New York" or "British Museum"
        """
        try:
            params = {
                "query": query,
                "language": language
            }
            
            if location:
                params["location"] = location
            if radius:
                params["radius"] = radius
            if min_price is not None:
                params["min_price"] = min_price
            if max_price is not None:
                params["max_price"] = max_price
            if open_now:
                params["open_now"] = True
                
            return self.client.places(**params)
        except Exception as e:
            raise GMapsAPIError(f"Text search error: {str(e)}")

    def place_details(self, place_id: str, language: str = "en", fields: Optional[List[str]] = None) -> Dict:
        """Get detailed information about a specific place."""
        try:
            return self.client.place(
                place_id,
                language=language,
                fields=fields
            )
        except Exception as e:
            raise GMapsAPIError(f"Place details error: {str(e)}")

    def nearby_search(
        self,
        location: Tuple[float, float],
        radius: int = 1000,
        keyword: Optional[str] = None,
        language: str = "en",
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        open_now: bool = False,
        place_type: Optional[str] = None
    ) -> Dict:
        """Search for places near a specific location."""
        try:
            params = {
                "location": location,
                "radius": radius,
                "language": language
            }
            
            if keyword:
                params["keyword"] = keyword
            if min_price is not None:
                params["min_price"] = min_price
            if max_price is not None:
                params["max_price"] = max_price
            if open_now:
                params["open_now"] = True
            if place_type:
                if place_type not in self.place_types:
                    raise ValueError(f"Invalid place type. Must be one of: {self.place_types}")
                params["type"] = place_type
                
            return self.client.places_nearby(**params)
        except Exception as e:
            raise GMapsAPIError(f"Nearby search error: {str(e)}")

    # Directions API Methods
    def _clean_directions_text(self, html_text: str) -> str:
        """Clean HTML from directions text and format it nicely."""
        # Remove HTML tags
        text = re.sub('<[^<]+?>', '', html_text)
        # Replace HTML entities
        text = text.replace('&nbsp;', ' ').replace('&lt;', '<').replace('&gt;', '>')
        # Replace /wbr/ with spaces
        text = text.replace('/wbr/', ' ')
        # Clean up multiple spaces
        text = ' '.join(text.split())
        # Format road references
        text = re.sub(r'(\w+)\s*/\s*(\w+)', r'\1/\2', text)  # Clean up road refs like "A1 / B2" to "A1/B2"
        
        return text

    def get_directions(
        self,
        origin: Union[str, Tuple[float, float]],
        destination: Union[str, Tuple[float, float]],
        mode: str = "driving",
        waypoints: Optional[List[str]] = None,
        alternatives: bool = False,
        avoid: Optional[List[str]] = None,
        language: str = "en",
        units: str = "metric",
        departure_time: Optional[Union[int, datetime]] = None,
        arrival_time: Optional[Union[int, datetime]] = None,
        optimize_waypoints: bool = False,
        transit_mode: Optional[List[str]] = None,
        transit_routing_preference: Optional[str] = None,
    ) -> Dict:
        """
        Get directions between locations.
        Modes: driving, walking, bicycling, transit
        """
        try:
            params = {
                "mode": mode,
                "language": language,
                "units": units,
                "alternatives": alternatives
            }
            
            if waypoints:
                if optimize_waypoints:
                    waypoints.insert(0, "optimize:true")
                params["waypoints"] = waypoints
                
            if avoid:
                params["avoid"] = "|".join(avoid)
                
            if mode == "transit":
                if departure_time:
                    params["departure_time"] = departure_time
                if arrival_time:
                    params["arrival_time"] = arrival_time
                if transit_mode:
                    params["transit_mode"] = transit_mode
                if transit_routing_preference:
                    params["transit_routing_preference"] = transit_routing_preference
            elif mode == "driving" and departure_time:
                params["departure_time"] = departure_time
                
            directions = self.client.directions(
                origin,
                destination,
                **params
            )

            # Clean up the directions text
            if directions and isinstance(directions, list):
                for route in directions:
                    if 'legs' in route:
                        for leg in route['legs']:
                            for step in leg['steps']:
                                step['clean_instructions'] = self._clean_directions_text(step['html_instructions'])

            return directions
        except Exception as e:
            raise GMapsAPIError(f"Directions error: {str(e)}")

    def distance_matrix(
        self,
        origins: List[Union[str, Tuple[float, float]]],
        destinations: List[Union[str, Tuple[float, float]]],
        mode: str = "driving",
        language: str = "en",
        avoid: Optional[List[str]] = None,
        units: str = "metric",
        departure_time: Optional[Union[int, datetime]] = None,
        arrival_time: Optional[Union[int, datetime]] = None,
        transit_mode: Optional[List[str]] = None,
        transit_routing_preference: Optional[str] = None,
    ) -> Dict:
        """Calculate distance and time between multiple origins and destinations."""
        try:
            params = {
                "mode": mode,
                "language": language,
                "units": units
            }
            
            if avoid:
                params["avoid"] = "|".join(avoid)
                
            if mode == "transit":
                if departure_time:
                    params["departure_time"] = departure_time
                if arrival_time:
                    params["arrival_time"] = arrival_time
                if transit_mode:
                    params["transit_mode"] = transit_mode
                if transit_routing_preference:
                    params["transit_routing_preference"] = transit_routing_preference
            elif mode == "driving" and departure_time:
                params["departure_time"] = departure_time
                
            return self.client.distance_matrix(
                origins,
                destinations,
                **params
            )
        except Exception as e:
            raise GMapsAPIError(f"Distance matrix error: {str(e)}")

    def get_timezone(
        self,
        location: Tuple[float, float],
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """Get timezone information for a location."""
        try:
            if not timestamp:
                timestamp = datetime.now()
            return self.client.timezone(location, timestamp)
        except Exception as e:
            raise GMapsAPIError(f"Timezone error: {str(e)}")

    # Elevation API
    def get_elevation(
        self,
        locations: List[Tuple[float, float]]
    ) -> List[Dict]:
        """Get elevation data for locations."""
        try:
            return self.client.elevation(locations)
        except Exception as e:
            raise GMapsAPIError(f"Elevation error: {str(e)}")

    # Static Maps API
    def get_static_map(
        self,
        center: Union[str, Tuple[float, float]],
        zoom: int = 13,
        size: Tuple[int, int] = (600, 400),
        maptype: str = "roadmap",
        markers: Optional[List[Dict]] = None,
        filename: Optional[str] = None
    ) -> str:
        """Get static map image."""
        try:
            base_url = "https://maps.googleapis.com/maps/api/staticmap"
            
            params = {
                "center": f"{center[0]},{center[1]}" if isinstance(center, tuple) else center,
                "zoom": zoom,
                "size": f"{size[0]}x{size[1]}",
                "maptype": maptype,
                "key": self.api_key
            }
            
            if markers:
                for idx, marker in enumerate(markers):
                    for key, value in marker.items():
                        params[f"markers[{idx}].{key}"] = value

            response = requests.get(base_url, params=params)
            
            if response.status_code == 200:
                if not filename:
                    filename = f"map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                filepath = self.output_dir / filename
                with open(filepath, "wb") as f:
                    f.write(response.content)
                return str(filepath)
            else:
                raise GMapsAPIError(f"Static map error: {response.status_code}")
        except Exception as e:
            raise GMapsAPIError(f"Static map error: {str(e)}")

    # Street View API
    def get_street_view(
        self,
        location: Union[str, Tuple[float, float]],
        size: Tuple[int, int] = (600, 400),
        heading: Optional[int] = None,
        pitch: Optional[int] = None,
        filename: Optional[str] = None
    ) -> str:
        """Get Street View image."""
        try:
            base_url = "https://maps.googleapis.com/maps/api/streetview"
            
            params = {
                "size": f"{size[0]}x{size[1]}",
                "key": self.api_key
            }
            
            if isinstance(location, tuple):
                params["location"] = f"{location[0]},{location[1]}"
            else:
                params["location"] = location
                
            if heading is not None:
                params["heading"] = heading
            if pitch is not None:
                params["pitch"] = pitch

            response = requests.get(base_url, params=params)
            
            if response.status_code == 200:
                if not filename:
                    filename = f"streetview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                filepath = self.output_dir / filename
                with open(filepath, "wb") as f:
                    f.write(response.content)
                return str(filepath)
            else:
                raise GMapsAPIError(f"Street View error: {response.status_code}")
        except Exception as e:
            raise GMapsAPIError(f"Street View error: {str(e)}") 