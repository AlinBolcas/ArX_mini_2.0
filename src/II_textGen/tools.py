import os
import sys
import json
import datetime
import webbrowser
import smtplib
import subprocess
from typing import Dict, List, Optional, Union, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define fallback classes for when imports fail
class DummyAPI:
    """Fallback class for when APIs aren't available"""
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        def dummy_method(*args, **kwargs):
            return None
        return dummy_method

# Fix imports with absolute paths for Streamlit deployment
try:
    # Import with standard project structure (works in most environments including Streamlit)
    from src.I_integrations.web_crawling import WebCrawler
    from src.I_integrations.replicate_API import ReplicateAPI
    from src.I_integrations.tripo_API import TripoAPI
    from src.I_integrations.openai_API import OpenAIAPI
except ImportError:
    # Try relative imports as fallback
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from I_integrations.web_crawling import WebCrawler
        from I_integrations.replicate_API import ReplicateAPI, download_file
        from I_integrations.tripo_API import TripoAPI
        from I_integrations.openai_API import OpenAIAPI
    except ImportError:
        # Fallback to dummy implementations
        WebCrawler = DummyAPI
        ReplicateAPI = DummyAPI
        TripoAPI = DummyAPI
        OpenAIAPI = DummyAPI

class Tools:
    """
    Integration toolkit providing streamlined access to web searching, media generation,
    and utility functions. Designed for both programmatic use and LLM tool calling.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, replicate_api_token: Optional[str] = None):
        """Initialize API clients with optional API key overrides."""
        # Initialize web crawler
        try:
            self.web_crawler = WebCrawler()
        except Exception as e:
            self.web_crawler = None
            print(f"Warning: Failed to initialize web crawler: {e}")
            
        # Initialize API clients with provided keys
        self.replicate = self._init_api(
            ReplicateAPI, 
            "Replicate",
            api_token=replicate_api_token
        )
        
        self.openai = self._init_api(
            OpenAIAPI, 
            "OpenAI",
            api_key=openai_api_key
        )
        
        # Initialize other APIs without keys
        self.tripo = self._init_api(TripoAPI, "Tripo3D")
        
        # Check email credentials
        self.email_available = bool(os.getenv("EMAIL_USER") and os.getenv("EMAIL_PASS"))
        if not self.email_available:
            print("Warning: Email credentials not found in environment variables.")
    
    def _init_api(self, api_class, api_name, **kwargs):
        """Helper to initialize API clients with error handling and optional API keys."""
        try:
            return api_class(**kwargs)
        except Exception as e:
            print(f"Warning: Failed to initialize {api_name} API: {e}")
            return None
    
    #################################
    # WEB SEARCH FUNCTIONS
    #################################
    
    def web_crawl(
        self,
        query: str,
        sources: str = "both",
        include_wiki_content: bool = False,
        max_wiki_sections: int = 3,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        Search the web using DuckDuckGo and/or Wikipedia.
        
        Args:
            query: Search query
            sources: Which sources to use ('ddg', 'wiki', or 'both')
            include_wiki_content: Whether to include full Wikipedia content
            max_wiki_sections: Maximum number of Wikipedia sections to include
            max_results: Maximum number of search results to return
            
        Returns:
            Dictionary with search results
        """
        results = {}
        
        if self.web_crawler is None:
            return {"error": "Web crawler not available"}
        
        try:
            if sources.lower() in ["ddg", "both"]:
                try:
                    ddg_results = self.web_crawler.search_ddg(query, max_results=max_results)
                    results["ddg_results"] = ddg_results
                except Exception as e:
                    results["ddg_error"] = f"DuckDuckGo search failed: {str(e)}"
                    print(f"DuckDuckGo search error (potentially rate limited): {e}")
            
            if sources.lower() in ["wiki", "both"]:
                try:
                    wiki_data = self.web_crawler.search_wikipedia(
                        query, 
                        include_content=include_wiki_content,
                        max_sections=max_wiki_sections
                    )
                    results["wiki_results"] = wiki_data
                except Exception as e:
                    results["wiki_error"] = f"Wikipedia search failed: {str(e)}"
                    print(f"Wikipedia search error: {e}")
            
            # If we got no results but no errors were recorded, add a generic error
            if not any(k for k in results.keys() if not k.endswith('_error')):
                results["error"] = "No search results found"
            
            return results
        
        except Exception as e:
            print(f"Error during web search: {e}")
            return {"error": f"Web search failed: {str(e)}"}
    
    #################################
    # MEDIA GENERATION FUNCTIONS
    #################################
    
    def generate_image(
        self,
        prompt: str,
        save_path: Optional[str] = None,
        safety_tolerance: int = 6  # Higher value = less strict
    ) -> str:
        """
        Generate an image from a text prompt using Replicate's Flux model.
        
        Args:
            prompt: Text description of the desired image
            save_path: Optional path to save the image (will auto-generate if None)
            safety_tolerance: Safety level (0 to 6, higher = less strict)
        
        Returns:
            str: Either:
            - Path to the saved image (if save successful)
            - URL to the generated image (if save failed)
            - Error message string if generation failed
        """
        if self.replicate is None or isinstance(self.replicate, DummyAPI):
            return "Error: Replicate API not available - imports failed"
        
        try:
            # Generate image using Flux
            print(f"Generating image with Flux using prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
            response = self.replicate.generate_image(
                prompt=prompt,
                aspect_ratio="1:1",  # Default square aspect ratio
                safety_tolerance=safety_tolerance
            )
            
            if not response:
                raise Exception("Replicate returned empty result")
            
            # Handle different types of responses that Replicate might return
            if hasattr(response, 'url'):
                image_url = response.url
            elif isinstance(response, str):
                image_url = response
            elif isinstance(response, list) and response:
                image_url = response[0]
            else:
                image_url = str(response)
            
            # Set up default save path if not provided
            if not save_path:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"image_{timestamp}.jpg"
                os.makedirs("output/images", exist_ok=True)
                save_path = os.path.join("output/images", filename)
            else:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Download image
            try:
                from urllib.request import urlretrieve
                urlretrieve(image_url, save_path)
                print(f"Image saved to: {save_path}")
                return save_path
            except Exception as e:
                print(f"Error saving image: {e}")
                return image_url
            
        except Exception as e:
            print(f"Error generating image with Flux: {e}")
            return f"Image generation failed: {str(e)}"
    
    def generate_music(
        self,
        prompt: str,
        duration: int = 10,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate music from a text prompt using Replicate API.
        
        Args:
            prompt: Text description of the desired music
            duration: Duration in seconds (default: 10)
            save_path: Optional path to save the music file
            
        Returns:
            Path to the saved music file or music URL if not saved
        """
        if not self.replicate:
            return "Error: Replicate API not available"
        
        try:
            # Generate music URL
            music_url = self.replicate.generate_music(
                prompt=prompt,
                duration=duration,
                model_version="stereo-large"
            )
            
            if not music_url:
                return "Failed to generate music"
                
            # Download if save path provided
            if save_path:
                try:
                    # Try to use the function from the correctly imported module
                    if hasattr(self.replicate, 'download_file'):
                        downloaded_path = self.replicate.download_file(music_url, output_dir="music", filename=save_path)
                    else:
                        # Otherwise use the one from the module directly if available 
                        from I_integrations.replicate_API import download_file
                        downloaded_path = download_file(music_url, output_dir="music", filename=save_path)
                    return downloaded_path or music_url
                except ImportError:
                    print("Warning: Could not import download_file function")
                    return music_url
            
            return music_url
            
        except Exception as e:
            print(f"Error generating music: {e}")
            return f"Music generation failed: {str(e)}"
    
    def generate_video(
        self,
        image_url: str,
        motion_prompt: str,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate video from an image using Replicate API.
        
        Args:
            image_url: URL of the image to animate
            motion_prompt: Text description of the desired motion
            save_path: Optional path to save the video
            
        Returns:
            Path to the saved video or video URL if not saved
        """
        if not self.replicate:
            return "Error: Replicate API not available"
        
        try:
            # Generate video URL
            video_url = self.replicate.generate_video(
                image_url=image_url,
                prompt=motion_prompt,
                num_frames=81  # Minimum required by the model
            )
            
            if not video_url:
                return "Failed to generate video"
                
            # Download if save path provided
            if save_path:
                from replicate_API import download_file
                downloaded_path = download_file(video_url, output_dir="videos", filename=save_path)
                return downloaded_path or video_url
            
            return video_url
            
        except Exception as e:
            print(f"Error generating video: {e}")
            return f"Video generation failed: {str(e)}"
    
    def generate_music_video(
        self,
        prompt: str,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a complete music video (image → video + music) from a text prompt.
        
        Args:
            prompt: Text description of the desired music video
            save_path: Optional base name for saved files
            
        Returns:
            Path to the saved video with audio or error message
        """
        if not self.replicate:
            return "Error: Replicate API not available"
        
        try:
            print("1. Generating image from prompt...")
            image_url = self.generate_image(prompt)
            if not image_url or image_url.startswith("Error") or image_url.startswith("Failed"):
                return f"Failed at image generation step: {image_url}"
                
            # Create slightly modified prompts for variety
            video_prompt = f"Camera slowly exploring {prompt}, with smooth movement"
            music_prompt = f"Soundtrack for {prompt}, emotionally fitting the visual scene"
            
            print("2. Generating video from image...")
            video_url = self.generate_video(image_url, video_prompt)
            if not video_url or video_url.startswith("Error") or video_url.startswith("Failed"):
                return f"Failed at video generation step: {video_url}"
                
            print("3. Generating music for video...")
            music_url = self.generate_music(music_prompt, duration=5)  # Short duration for demo
            if not music_url or music_url.startswith("Error") or music_url.startswith("Failed"):
                return f"Failed at music generation step: {music_url}"
            
            # Download files
            from replicate_API import download_file, merge_video_audio
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = download_file(video_url, output_dir="videos", filename=f"{save_path or 'music_video'}_video_{timestamp}.mp4")
            music_path = download_file(music_url, output_dir="music", filename=f"{save_path or 'music_video'}_audio_{timestamp}.mp3")
            
            if not video_path or not music_path:
                return f"Failed to download generated files. Video URL: {video_url}, Music URL: {music_url}"
            
            # Merge video and audio
            print("4. Merging video and audio...")
            merged_filename = f"{save_path or 'music_video'}_merged_{timestamp}.mp4"
            merged_path = merge_video_audio(video_path, music_path, filename=merged_filename)
            
            if merged_path:
                return merged_path
            else:
                return f"Video: {video_path}, Music: {music_path} (Merge failed)"
            
        except Exception as e:
            print(f"Error generating music video: {e}")
            return f"Music video generation failed: {str(e)}"
    
    def generate_threed(
        self,
        prompt: Optional[str] = None, 
        image_url: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate 3D model from text or image using Tripo3D API.
        
        Args:
            prompt: Text description for text-to-3D (required if no image_url)
            image_url: URL to image for image-to-3D (required if no prompt)
            save_path: Optional path to save the 3D model
            
        Returns:
            Path to the saved 3D model or error message
        """
        if not self.tripo:
            return "Error: Tripo3D API not available"
            
        if not prompt and not image_url:
            return "Error: Either prompt or image_url must be provided"
        
        try:
            import asyncio
            
            # Set output path if not provided
            if not save_path:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = "threed_" + timestamp + ".glb"
                save_dir = os.path.join("output", "threed")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, model_name)
            
            # Run the async generation in a new event loop
            return asyncio.run(self.tripo.generate_threed(
                prompt=prompt,
                image_url=image_url,
                output_path=save_path
            ))
            
        except Exception as e:
            print(f"Error generating 3D model: {e}")
            return f"3D model generation failed: {str(e)}"
    
    #################################
    # UTILITY FUNCTIONS
    #################################
    
    def get_current_datetime(self, format: str = "iso") -> str:
        """
        Get the current date and time in the specified format.
        
        Args:
            format: Format type ("iso", "human", "date", "time")
            
        Returns:
            Formatted datetime string
        """
        now = datetime.datetime.now()
        
        if format.lower() == "iso":
            return now.isoformat()
        elif format.lower() == "human":
            return now.strftime("%A, %B %d, %Y at %I:%M %p")
        elif format.lower() == "date":
            return now.strftime("%Y-%m-%d")
        elif format.lower() == "time":
            return now.strftime("%H:%M:%S")
        else:
            return now.isoformat()
    
    def open_url_in_browser(self, url: str) -> bool:
        """
        Open a URL in the default web browser.
        
        Args:
            url: The URL to open
            
        Returns:
            True if successful, False otherwise
        """
        try:
            webbrowser.open(url)
            return True
        except Exception as e:
            print(f"Error opening URL: {e}")
            return False
    
    def send_email(
        self,
        recipient: str,
        subject: str,
        body: str,
        sender: Optional[str] = None,
        html: bool = False
    ) -> bool:
        """
        Send an email using configured SMTP settings.
        
        Args:
            recipient: Recipient email address
            subject: Email subject
            body: Email body content
            sender: Optional sender (uses EMAIL_USER env var if not provided)
            html: Whether the body contains HTML
            
        Returns:
            True if successful, False otherwise
        """
        if not self.email_available:
            print("Email credentials not configured in environment variables")
            return False
        
        try:
            # Get credentials from environment
            email_user = sender or os.getenv("EMAIL_USER")
            email_pass = os.getenv("EMAIL_PASS")
            
            # Create message
            msg = MIMEMultipart()
            msg["From"] = email_user
            msg["To"] = recipient
            msg["Subject"] = subject
            
            # Attach body with appropriate type
            content_type = "html" if html else "plain"
            msg.attach(MIMEText(body, content_type))
            
            # Setup and send email
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(email_user, email_pass)
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Error sending email: {e}")
            return False

    #################################
    # WEATHER FUNCTIONS
    #################################

    def get_weather(
        self,
        location: str,
        units: str = "metric"
    ) -> Dict[str, Any]:
        """
        Get current weather for a location using OpenWeatherMap API.
        
        Args:
            location: Location string (e.g., "London,UK", "New York,US")
            units: Unit system - "metric" (Celsius) or "imperial" (Fahrenheit)
            
        Returns:
            Dictionary with weather information
        """
        try:
            # Lazy import to avoid circular dependencies
            from openweather_API import OpenWeatherAPI
            
            # Initialize API if needed
            if not hasattr(self, "weather_api"):
                self.weather_api = OpenWeatherAPI()
            
            # Get current weather
            weather_data = self.weather_api.get_current_weather(location, units=units)
            
            # Format the result
            if "main" in weather_data and "weather" in weather_data:
                # Extract key information
                formatted_result = {
                    "location": weather_data.get("name", location),
                    "country": weather_data.get("sys", {}).get("country", ""),
                    "temperature": weather_data["main"].get("temp"),
                    "feels_like": weather_data["main"].get("feels_like"),
                    "humidity": weather_data["main"].get("humidity"),
                    "pressure": weather_data["main"].get("pressure"),
                    "wind_speed": weather_data.get("wind", {}).get("speed"),
                    "description": weather_data["weather"][0].get("description") if weather_data["weather"] else "",
                    "condition": weather_data["weather"][0].get("main") if weather_data["weather"] else "",
                    "icon": weather_data["weather"][0].get("icon") if weather_data["weather"] else "",
                    "units": units,
                    "timestamp": weather_data.get("dt"),
                    "timezone": weather_data.get("timezone"),
                    "raw_data": weather_data  # Include raw data for advanced usage
                }
                return formatted_result
            else:
                return {"error": "Weather data not available", "raw_data": weather_data}
            
        except Exception as e:
            print(f"Error getting weather: {e}")
            return {"error": f"Weather retrieval failed: {str(e)}"}

    def get_forecast(
        self,
        location: str,
        days: int = 5,
        units: str = "metric"
    ) -> Dict[str, Any]:
        """
        Get weather forecast for a location using OpenWeatherMap API.
        
        Args:
            location: Location string (e.g., "London,UK", "New York,US")
            days: Number of days for forecast (up to 5)
            units: Unit system - "metric" (Celsius) or "imperial" (Fahrenheit)
            
        Returns:
            Dictionary with forecast information
        """
        try:
            # Lazy import to avoid circular dependencies
            from openweather_API import OpenWeatherAPI
            
            # Initialize API if needed
            if not hasattr(self, "weather_api"):
                self.weather_api = OpenWeatherAPI()
            
            # Get forecast
            forecast_data = self.weather_api.get_forecast(location, units=units, days=days)
            
            # Format the result
            if "list" in forecast_data:
                # Extract forecast entries
                entries = forecast_data["list"]
                city_info = forecast_data.get("city", {})
                
                # Group by day and extract key information
                days_forecast = {}
                
                for entry in entries:
                    # Get date from dt_txt (format: "2023-01-01 12:00:00")
                    if "dt_txt" in entry:
                        date_str = entry["dt_txt"].split()[0]  # Get just the date part
                        
                        if date_str not in days_forecast:
                            days_forecast[date_str] = []
                        
                        # Extract key information
                        forecast_entry = {
                            "time": entry["dt_txt"].split()[1],  # Get just the time part
                            "temperature": entry.get("main", {}).get("temp"),
                            "feels_like": entry.get("main", {}).get("feels_like"),
                            "humidity": entry.get("main", {}).get("humidity"),
                            "description": entry.get("weather", [{}])[0].get("description") if entry.get("weather") else "",
                            "condition": entry.get("weather", [{}])[0].get("main") if entry.get("weather") else "",
                            "icon": entry.get("weather", [{}])[0].get("icon") if entry.get("weather") else "",
                            "wind_speed": entry.get("wind", {}).get("speed"),
                        }
                        
                        days_forecast[date_str].append(forecast_entry)
                
                return {
                    "location": city_info.get("name", location),
                    "country": city_info.get("country", ""),
                    "timezone": city_info.get("timezone"),
                    "days": days_forecast,
                    "units": units,
                    "raw_data": forecast_data  # Include raw data for advanced usage
                }
            else:
                return {"error": "Forecast data not available", "raw_data": forecast_data}
            
        except Exception as e:
            print(f"Error getting forecast: {e}")
            return {"error": f"Forecast retrieval failed: {str(e)}"}

    def text_to_speech(
        self,
        text: str,
        voice: str = "alloy",
        output_path: Optional[str] = None,
        speed: float = 1.0
    ) -> str:
        """
        Convert text to speech using OpenAI's TTS API.
        
        Args:
            text: Text content to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer, ash, sage, coral)
            output_path: Path to save the audio file
            speed: Speech speed (0.25 to 4.0)
            
        Returns:
            Path to the saved audio file
        """
        if not self.openai:
            return "Error: OpenAI API not available"
        
        # Validate voice parameter against OpenAI's allowed voices
        allowed_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "ash", "sage", "coral"]
        if voice not in allowed_voices:
            print(f"Warning: '{voice}' is not a valid OpenAI voice. Using 'alloy' instead.")
            voice = "alloy"
        
        try:
            # Call OpenAI's text-to-speech API
            audio_path = self.openai.text_to_speech(
                text=text,
                voice=voice,
                output_path=output_path,
                speed=speed
            )
            
            return audio_path
        except Exception as e:
            print(f"Error generating speech: {e}")
            return f"Speech generation failed: {str(e)}"

    def transcribe_speech(
        self,
        audio_file: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe speech from an audio file using OpenAI's Whisper API.
        
        Args:
            audio_file: Path to the audio file
            language: Optional language code (e.g., 'en', 'fr')
            prompt: Optional prompt to guide transcription
            
        Returns:
            Dictionary with transcription text and metadata
        """
        if not self.openai:
            return {"error": "OpenAI API not available"}
        
        try:
            # Call OpenAI's transcription API
            result = self.openai.transcribe_audio(
                audio_file=audio_file,
                language=language,
                prompt=prompt
            )
            
            return result
        except Exception as e:
            print(f"Error transcribing speech: {e}")
            return {"error": f"Transcription failed: {str(e)}"}


# Example usage
if __name__ == "__main__":
    print("\n===== TOOLS DEMO =====\n")
    tools = Tools()
    
    # Function to prompt for input with default value
    def prompt(message, default=None):
        result = input(f"{message} [{default}]: ") if default else input(f"{message}: ")
        return result if result.strip() else default
    
    # Demo menu
    while True:
        print("\nAvailable Tools:")
        print("1. Web Search")
        print("2. Wikipedia")
        print("3. Generate Image")
        print("4. Generate Music")
        print("5. Generate Video (from image URL)")
        print("6. Generate Music Video (full pipeline)")
        print("7. Generate 3D Model")
        print("8. Get Current DateTime")
        print("9. Open URL in Browser")
        print("10. Send Email")
        print("11. Get Weather")
        print("12. Get Forecast")
        print("13. Text-to-Speech")
        print("14. Speech-to-Text")
        print("0. Exit")
        
        choice = prompt("Select a tool to demo", "0")
        
        if choice == "0":
            print("Exiting demo.")
            break
            
        elif choice == "1":
            query = prompt("Enter search query", "AI advancements")
            results = tools.web_crawl(query, sources="ddg")
            print("\nSearch Results:")
            if "ddg_results" in results:
                for i, result in enumerate(results["ddg_results"][:3], 1):
                    print(f"\n{i}. {result.get('title', '')}")
                    print(f"   URL: {result.get('link', '')}")
                    print(f"   {result.get('snippet', '')[:100]}...")
            else:
                print("No results found")
                
        elif choice == "2":
            query = prompt("Enter Wikipedia search query", "Machine learning")
            get_content = prompt("Get full content? (y/n)", "n").lower() == "y"
            result = tools.web_crawl(query, sources="wiki", include_wiki_content=get_content)
            
            if "wiki_results" in result:
                wiki_data = result["wiki_results"]
                print(f"\nWikipedia: {wiki_data.get('title', '')}")
                print(f"\nSummary: {wiki_data.get('summary', '')[:300]}...")
                if get_content and "content" in wiki_data:
                    print(f"\nContent preview: {wiki_data.get('content', '')[:300]}...")
            else:
                print("No Wikipedia results found")
                
        elif choice == "3":
            prompt_text = prompt("Enter image description", "A serene landscape with mountains and a lake at sunset")
            print("\nGenerating image with Flux, please wait...")
            result = tools.generate_image(prompt_text)
            print(f"Result: {result}")
            
            # Open the image if saved successfully
            if os.path.exists(result):
                try:
                    if sys.platform == "darwin":  # macOS
                        # Try QuickLook first
                        subprocess.run(["qlmanage", "-p", result], 
                                      stdout=subprocess.DEVNULL, 
                                      stderr=subprocess.DEVNULL)
                    elif sys.platform == "win32":  # Windows
                        os.startfile(result)
                    else:  # Linux
                        subprocess.run(["xdg-open", result])
                    print("Image opened for preview")
                except Exception as e:
                    print(f"Couldn't open image for preview: {e}")
            
        elif choice == "4":
            prompt_text = prompt("Enter music description", "Epic orchestral music with soaring strings and dramatic percussion")
            duration = int(prompt("Duration in seconds", "10"))
            print("\nGenerating music, please wait...")
            result = tools.generate_music(prompt_text, duration=duration)
            print(f"Result: {result}")
            
        elif choice == "5":
            image_url = prompt("Enter image URL", "")
            motion = prompt("Enter motion description", "Camera slowly panning around the scene, revealing details")
            if not image_url:
                print("Image URL is required for video generation.")
                continue
            print("\nGenerating video, please wait...")
            result = tools.generate_video(image_url, motion)
            print(f"Result: {result}")
            
        elif choice == "6":
            prompt_text = prompt("Enter music video concept", "A cosmic journey through nebulae and star formations")
            print("\nGenerating complete music video (this may take several minutes)...")
            result = tools.generate_music_video(prompt_text)
            print(f"Result: {result}")
            
        elif choice == "7":
            prompt_type = prompt("Generate from text or image? (text/image)", "text").lower()
            
            if prompt_type.startswith("t"):
                prompt_text = prompt("Enter 3D model description", "A futuristic spacecraft with sleek design")
                print("\nGenerating 3D model from text, please wait...")
                result = tools.generate_threed(prompt=prompt_text)
            else:
                image_url = prompt("Enter image URL", "")
                if not image_url:
                    print("Image URL is required for image-to-3D generation.")
                    continue
                print("\nGenerating 3D model from image, please wait...")
                result = tools.generate_threed(image_url=image_url)
                
            print(f"Result: {result}")
            
        elif choice == "8":
            format_type = prompt("Format (iso/human/date/time)", "human")
            result = tools.get_current_datetime(format_type)
            print(f"Current datetime: {result}")
            
        elif choice == "9":
            url = prompt("Enter URL to open", "https://www.google.com")
            tools.open_url_in_browser(url)
            print(f"Opening {url} in browser...")
            
        elif choice == "10":
            if not tools.email_available:
                print("Email is not configured. Add EMAIL_USER and EMAIL_PASS to .env file.")
                continue
                
            recipient = prompt("Enter recipient email")
            subject = prompt("Enter subject", "Test email from Tools")
            body = prompt("Enter message", "This is a test email sent from the Tools module.")
            
            if not recipient:
                print("Recipient email is required.")
                continue
                
            print("\nSending email...")
            result = tools.send_email(recipient, subject, body)
            if result:
                print("Email sent successfully!")
            else:
                print("Failed to send email.")
                
        elif choice == "11":
            location = prompt("Enter location (e.g., 'London,UK', 'New York,US')", "London,UK")
            units = prompt("Units (metric/imperial)", "metric")
            forecast = prompt("Get forecast? (y/n)", "n").lower() == "y"
            
            if forecast:
                days = int(prompt("Number of forecast days (1-5)", "3"))
                print(f"\nGetting {days}-day forecast for {location}...")
                result = tools.get_forecast(location, days=days, units=units)
                
                if "error" not in result:
                    print(f"\nForecast for {result.get('location')}, {result.get('country')}:")
                    for date, entries in result.get("days", {}).items():
                        print(f"\n{date}:")
                        for entry in entries[:3]:  # Show first 3 time slots per day
                            print(f"  {entry.get('time')}: {entry.get('temperature')}°{' C' if units=='metric' else ' F'}, {entry.get('description')}")
                        if len(entries) > 3:
                            print(f"  ... and {len(entries) - 3} more time slots")
                else:
                    print(f"Error: {result.get('error')}")
            else:
                print(f"\nGetting current weather for {location}...")
                result = tools.get_weather(location, units=units)
                
                if "error" not in result:
                    temp_unit = "C" if units == "metric" else "F"
                    speed_unit = "m/s" if units == "metric" else "mph"
                    
                    print(f"\nCurrent weather for {result.get('location')}, {result.get('country')}:")
                    print(f"Temperature: {result.get('temperature')}°{temp_unit}")
                    print(f"Feels like: {result.get('feels_like')}°{temp_unit}")
                    print(f"Condition: {result.get('description')}")
                    print(f"Humidity: {result.get('humidity')}%")
                    print(f"Wind speed: {result.get('wind_speed')} {speed_unit}")
                else:
                    print(f"Error: {result.get('error')}")
                
        elif choice == "12":
            location = prompt("Enter location", "London,UK")
            days = int(prompt("Enter number of days for forecast", "5"))
            units = prompt("Enter units (metric/imperial)", "metric")
            result = tools.get_forecast(location, days, units)
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print("\nForecast Information:")
                for key, value in result.items():
                    if key != "raw_data":
                        print(f"{key.capitalize()}:")
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                print(f"  {subkey.capitalize()}: {subvalue}")
                        else:
                            print(f"  {value}")
                
        elif choice == "13":
            text = prompt("Enter text to convert to speech", "Welcome to the demo of our text to speech capability.")
            voice = prompt("Choose voice (alloy, echo, fable, onyx, nova, shimmer)", "alloy")
            output_path = prompt("Enter output file name (optional)", "demo_speech.mp3")
            
            print("\nGenerating speech...")
            result = tools.text_to_speech(text, voice, output_path)
            print(f"Speech generated and saved to: {result}")
            
        elif choice == "14":
            audio_file = prompt("Enter path to audio file", "")
            if not audio_file:
                print("Audio file path is required.")
                continue
            
            language = prompt("Enter language code (optional, e.g., 'en')", "")
            guide_prompt = prompt("Enter guiding prompt (optional)", "")
            
            print("\nTranscribing speech...")
            result = tools.transcribe_speech(
                audio_file=audio_file,
                language=language if language else None,
                prompt=guide_prompt if guide_prompt else None
            )
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print("\nTranscription:")
                print(result.get("text", "No text returned"))
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")
    
    print("\n===== DEMO COMPLETE =====") 