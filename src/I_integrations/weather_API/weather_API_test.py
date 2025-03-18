import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json
from typing import Dict

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import utilities using FileFinder
from modules.VIII_utils.file_finder import FileFinder
finder = FileFinder()

# Import required modules and functions
OpenWeatherAPI = finder.get_class('openweather_API.py', 'OpenWeatherAPI')
TomorrowAPI = finder.get_class('tomorrow_API.py', 'TomorrowAPI')
utils = finder.import_module('utils.py')
printColoured = utils.printColoured

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

def compare_weather_data(ow_data: Dict, tm_data: Dict) -> None:
    """Compare weather data from both services."""
    logger.info(f"\n{printColoured('🔄 Weather Data Comparison:', 'white')}")
    logger.info(printColoured("-" * 30, 'white'))
    
    # OpenWeather data
    ow_temp = ow_data['main']['temp']
    ow_humidity = ow_data['main']['humidity']
    ow_wind = ow_data['wind']['speed']
    ow_desc = ow_data['weather'][0]['description']
    
    # Tomorrow.io data
    tm_temp = tm_data['data']['values']['temperature']
    tm_humidity = tm_data['data']['values']['humidity']
    tm_wind = tm_data['data']['values']['windSpeed']
    
    # Print comparison
    logger.info(f"Temperature:")
    logger.info(f"  OpenWeather: {ow_temp}°C")
    logger.info(f"  Tomorrow.io: {tm_temp}°C")
    logger.info(f"  Difference: {abs(ow_temp - tm_temp):.1f}°C")
    
    logger.info(f"\nHumidity:")
    logger.info(f"  OpenWeather: {ow_humidity}%")
    logger.info(f"  Tomorrow.io: {tm_humidity}%")
    logger.info(f"  Difference: {abs(ow_humidity - tm_humidity):.1f}%")
    
    logger.info(f"\nWind Speed:")
    logger.info(f"  OpenWeather: {ow_wind} m/s")
    logger.info(f"  Tomorrow.io: {tm_wind} m/s")
    logger.info(f"  Difference: {abs(ow_wind - tm_wind):.1f} m/s")
    
    logger.info(f"\nOpenWeather Description: {ow_desc}")

def main():
    """Run weather API tests."""
    logger.info("\n🌤️ Starting Weather API Tests")
    logger.info("===========================")
    
    try:
        # Initialize wrappers
        openweather = OpenWeatherAPI()
        tomorrow = TomorrowAPI()
        
        # Test locations
        city = "London"
        lat, lon = 51.5074, -0.1278
        location = (lat, lon)
        
        # Current Weather Comparison
        logger.info(f"\n{printColoured('📍 Location: London, UK', 'white')}")
        
        # OpenWeather current
        logger.info(f"\n{printColoured('☀️ OpenWeather Current:', 'cyan')}")
        ow_current = openweather.get_current_weather(city)
        logger.info(f"🌡️ Temperature: {ow_current['main']['temp']}°C")
        logger.info(f"💨 Wind: {ow_current['wind']['speed']} m/s")
        logger.info(f"☁️ Conditions: {ow_current['weather'][0]['description']}")
        
        # Tomorrow.io current
        logger.info(f"\n{printColoured('🌤️ Tomorrow.io Current:', 'yellow')}")
        tm_current = tomorrow.get_realtime(location)
        logger.info(f"🌡️ Temperature: {tm_current['data']['values']['temperature']}°C")
        logger.info(f"💨 Wind: {tm_current['data']['values']['windSpeed']} m/s")
        logger.info(f"💧 Humidity: {tm_current['data']['values']['humidity']}%")
        
        # Compare current weather
        compare_weather_data(ow_current, tm_current)
        
        # Forecast Comparison
        logger.info(f"\n{printColoured('📅 24-Hour Forecast Comparison:', 'white')}")
        
        # OpenWeather forecast
        logger.info(f"\n{printColoured('☀️ OpenWeather Forecast:', 'cyan')}")
        ow_forecast = openweather.get_forecast(city, days=1)
        for item in ow_forecast['list'][:8]:  # 24 hours (3-hour steps)
            time = datetime.fromtimestamp(item['dt']).strftime('%H:%M')
            logger.info(f"⏰ {time}: {item['main']['temp']}°C, {item['weather'][0]['description']}")
        
        # Tomorrow.io forecast
        logger.info(f"\n{printColoured('🌤️ Tomorrow.io Forecast:', 'yellow')}")
        tm_forecast = tomorrow.get_forecast(location, timesteps=["1h"])
        for item in tm_forecast['timelines']['hourly'][:24]:  # 24 hours
            time = datetime.fromisoformat(item['time']).strftime('%H:%M')
            logger.info(f"⏰ {time}: {item['values']['temperature']}°C")
        
        logger.info(f"\n{printColoured('✨ All Tests Complete', 'white')}")
        
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 