import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List
from pprint import pformat
import re

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import utilities using FileFinder
from modules.VIII_utils.file_finder import FileFinder
finder = FileFinder()

# Import required modules and functions
GMapsAPI = finder.get_class('gmaps_API.py', 'GMapsAPI')
utils = finder.import_module('utils.py')
printColoured = utils.printColoured
quick_look = utils.quick_look

# Custom formatter for prettier logging
class PrettyFormatter(logging.Formatter):
    def format(self, record):
        if 'Testing' in record.msg:
            return f"\n{printColoured('>> ' + record.msg, 'magenta')}"
        elif 'Response:' in record.msg:
            return f"\n{printColoured('Response:', 'yellow')}\n{record.msg.replace('Response:', '')}"
        elif 'Error:' in record.msg:
            return f"\n{printColoured('❌ ' + record.msg, 'red')}"
        elif 'Success:' in record.msg:
            return f"\n{printColoured('✓ ' + record.msg, 'green')}"
        else:
            return f"{printColoured('>', 'blue')} {record.msg}"

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(PrettyFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def format_place(place: Dict) -> None:
    """Format and print place details."""
    logger.info(f"\n🏢 {printColoured(place.get('name', 'Unknown'), 'white')}")
    if place.get('formatted_address'):
        logger.info(f"📍 Address: {place['formatted_address']}")
    if place.get('rating'):
        logger.info(f"⭐ Rating: {place['rating']} ({place.get('user_ratings_total', 0)} reviews)")
    if place.get('formatted_phone_number'):
        logger.info(f"📞 Phone: {place['formatted_phone_number']}")
    if place.get('website'):
        logger.info(f"🌐 Website: {place['website']}")
    if place.get('opening_hours', {}).get('weekday_text'):
        logger.info(f"🕒 Hours: {place['opening_hours']['weekday_text'][0]}")

def format_directions(directions: List[Dict]) -> None:
    """Format and print directions details."""
    if not directions or not isinstance(directions, list):
        logger.error("No directions found")
        return
    
    # Primary route
    route = directions[0]
    if 'legs' not in route:
        logger.error("Invalid route format")
        return
        
    leg = route['legs'][0]
    logger.info(f"\n🚗 {printColoured('Primary Route:', 'white')}")
    logger.info(f"📍 From: {leg['start_address']}")
    logger.info(f"🎯 To: {leg['end_address']}")
    logger.info(f"⏱️ Total Duration: {leg['duration']['text']}")
    logger.info(f"📏 Total Distance: {leg['distance']['text']}")
    
    logger.info(f"\n📝 Navigation Steps:")
    for i, step in enumerate(leg['steps'], 1):
        details = f"({step['distance']['text']}"
        if 'duration' in step:
            details += f" - {step['duration']['text']}"
        details += ")"
        
        logger.info(f"{i}. {step['clean_instructions']} {details}")
        
    # Alternative routes
    if len(directions) > 1:
        logger.info(f"\n🔄 {printColoured('Alternative Routes:', 'white')}")
        for i, route in enumerate(directions[1:], 2):
            leg = route['legs'][0]
            logger.info(f"\nRoute {i}:")
            logger.info(f"⏱️ Duration: {leg['duration']['text']}")
            logger.info(f"📏 Distance: {leg['distance']['text']}")
            logger.info("Key steps:")
            # Show first 2 significant turns
            shown_steps = 0
            for step in leg['steps']:
                if 'maneuver' in step:
                    logger.info(f"  • {step['clean_instructions']} ({step['distance']['text']})")
                    shown_steps += 1
                    if shown_steps >= 2:
                        break

def main():
    """Run Google Maps API test suite."""
    logger.info("\n🗺️ Starting Google Maps API Test Suite")
    logger.info("===================================")
    
    try:
        maps = GMapsAPI()
        target_address = "2 Empingham Road, Stamford, UK"
        
        # Test 1: Text Search
        logger.info("\nTesting Text Search")
        try:
            results = maps.text_search(
                query=target_address,
                language="en"
            )
            
            if results.get('results'):
                logger.info(f"Found {len(results['results'])} results")
                place = results['results'][0]
                format_place(place)
                test_place_id = place['place_id']
                
                # Save coordinates for other tests
                location = place['geometry']['location']
                test_coords = (location['lat'], location['lng'])
                
                logger.info("\nSuccess: Text search completed")
            else:
                logger.error("No results found")
                
        except Exception as e:
            logger.error(f"Text search failed: {str(e)}")
            
        # Test 2: Place Details
        logger.info("\nTesting Place Details")
        try:
            if 'test_place_id' in locals():
                details = maps.place_details(
                    place_id=test_place_id,
                    fields=['name', 'formatted_address', 'rating', 'website', 
                           'formatted_phone_number', 'opening_hours', 'geometry']
                )
                
                if details.get('result'):
                    format_place(details['result'])
                    logger.info("\nSuccess: Place details retrieved")
                else:
                    logger.error("No details found")
                    
        except Exception as e:
            logger.error(f"Place details failed: {str(e)}")
            
        # Test 3: Nearby Search
        logger.info("\nTesting Nearby Search")
        try:
            if 'test_coords' in locals():
                results = maps.nearby_search(
                    location=test_coords,
                    radius=1000,  # 1km radius
                    keyword="restaurant",  # Looking for nearby restaurants
                    open_now=True
                )
                
                if results.get('results'):
                    logger.info(f"Found {len(results['results'])} nearby restaurants")
                    for place in results['results'][:3]:  # Show top 3
                        format_place(place)
                    logger.info("\nSuccess: Nearby search completed")
                else:
                    logger.error("No nearby places found")
                    
        except Exception as e:
            logger.error(f"Nearby search failed: {str(e)}")
            
        # Test 4: Directions
        logger.info("\nTesting Directions")
        try:
            # Directions to nearest train station
            directions = maps.get_directions(
                origin=target_address,
                destination="Stamford Railway Station",
                mode="driving",
                alternatives=True,
                language="en"
            )
            
            if directions and isinstance(directions, list):
                format_directions(directions[0])  # Show primary route
                if len(directions) > 1:
                    logger.info(f"\n🔄 Alternative routes:")
                    for i, route in enumerate(directions[1:], 2):
                        leg = route['legs'][0]
                        logger.info(f"Route {i}:")
                        logger.info(f"  ⏱️ Duration: {leg['duration']['text']}")
                        logger.info(f"  📏 Distance: {leg['distance']['text']}")
                        logger.info(f"  📍 Via: {' → '.join([step['clean_instructions'] for step in leg['steps'][:2]])}")
                logger.info("\nSuccess: Directions retrieved")
            else:
                logger.error(f"Invalid directions response: {directions}")
                
        except Exception as e:
            logger.error(f"Directions failed: {str(e)}")
            
        # Test 5: Distance Matrix
        logger.info("\nTesting Distance Matrix")
        try:
            # Distance to nearby points of interest
            destinations = ["Stamford Railway Station", 
                          "Burghley House",
                          "Stamford Town Centre"]
            
            matrix = maps.distance_matrix(
                origins=[target_address],
                destinations=destinations,
                mode="driving"
            )
            
            if matrix.get('rows'):
                logger.info(f"\n📊 {printColoured('Distance Matrix Results:', 'white')}")
                for i, element in enumerate(matrix['rows'][0]['elements']):
                    logger.info(f"To {destinations[i]}:")
                    logger.info(f"  ⏱️ Duration: {element['duration']['text']}")
                    logger.info(f"  📏 Distance: {element['distance']['text']}")
                logger.info("\nSuccess: Distance matrix calculated")
            else:
                logger.error("No matrix results found")
                
        except Exception as e:
            logger.error(f"Distance matrix failed: {str(e)}")
        
        # Test 6: Timezone
        logger.info("\nTesting Timezone API")
        try:
            if 'test_coords' in locals():
                result = maps.get_timezone(test_coords)
                if result:
                    logger.info(f"Timezone: {result.get('timeZoneId')}")
                    logger.info(f"Timezone name: {result.get('timeZoneName')}")
                    logger.info("\nSuccess: Timezone retrieved")
                else:
                    logger.error("Timezone lookup failed")
        except Exception as e:
            logger.error(f"Timezone lookup failed: {str(e)}")

        # Test 7: Elevation
        logger.info("\nTesting Elevation API")
        try:
            if 'test_coords' in locals():
                result = maps.get_elevation([test_coords])
                if result:
                    logger.info(f"Elevation: {result[0].get('elevation')} meters")
                    logger.info("\nSuccess: Elevation data retrieved")
                else:
                    logger.error("Elevation lookup failed")
        except Exception as e:
            logger.error(f"Elevation lookup failed: {str(e)}")

        # Test 8: Static Map
        logger.info("\nTesting Static Map API")
        try:
            if 'test_coords' in locals():
                result = maps.get_static_map(
                    center=test_coords,
                    zoom=16,  # Closer zoom for residential area
                    size=(800, 600),
                    markers=[{"location": target_address}]
                )
                if result:
                    logger.info(f"Static map saved to: {result}")
                    quick_look(result)
                    logger.info("\nSuccess: Static map generated")
                else:
                    logger.error("Static map generation failed")
        except Exception as e:
            logger.error(f"Static map generation failed: {str(e)}")

        # Test 9: Street View
        logger.info("\nTesting Street View API")
        try:
            result = maps.get_street_view(
                location=target_address,
                size=(800, 600),
                heading=90,
                pitch=0
            )
            if result:
                logger.info(f"Street view saved to: {result}")
                quick_look(result)
                logger.info("\nSuccess: Street view generated")
            else:
                logger.error("Street view generation failed")
        except Exception as e:
            logger.error(f"Street view generation failed: {str(e)}")

        logger.info(f"\n{printColoured('✨ Test Suite Complete', 'white')}")
        
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 