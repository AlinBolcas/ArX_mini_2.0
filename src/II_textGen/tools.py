import os
import sys
import json
import datetime
import webbrowser
import smtplib
import subprocess
import logging # Import logging
from typing import Dict, List, Optional, Union, Any, Literal
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to the Python path to enable absolute imports
# This makes imports work regardless of how the script is run (e.g., directly or from root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup initial logger for import errors
initial_logger = logging.getLogger(__name__ + "_setup")
initial_logger.addHandler(logging.StreamHandler()) # Ensure setup errors are visible
initial_logger.setLevel(logging.INFO)

# Now attempt the absolute imports
try:
    from src.I_integrations.web_crawling import WebCrawler
    from src.I_integrations.replicate_API import ReplicateAPI
    from src.I_integrations.openai_API import OpenAIWrapper
    from src.I_integrations.news_API import News
except ImportError as e:
    initial_logger.error(f"Error during absolute import: {e}")
    initial_logger.info("Attempting relative imports as a fallback...")
    # Fallback to relative imports (might work in some structures, less reliable)
    try:
        # Adjust path for relative imports if needed when running directly
        if 'src' not in sys.path[0]: # Check if src/ path needs to be added
             sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        
        from I_integrations.web_crawling import WebCrawler
        from I_integrations.replicate_API import ReplicateAPI
        from I_integrations.openai_API import OpenAIWrapper
        from I_integrations.news_API import News
        initial_logger.info("Relative imports successful.")
    except ImportError as final_e:
        initial_logger.error(f"Error during relative import fallback: {final_e}")
        raise ImportError("Failed to import required modules using both absolute and relative paths.") from final_e

# Set up module-level logger
logger = logging.getLogger(__name__)
# Ensure the logger has a handler if not configured by the root logger
# This prevents "No handlers could be found for logger..." messages
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler()) # Add console handler
    logger.setLevel(logging.INFO) # Default level if not set by calling module

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
            logger.warning(f"Failed to initialize web crawler: {e}")
            
        # Initialize API clients with provided keys
        self.replicate = self._init_api(
            ReplicateAPI, 
            "Replicate",
            api_token=replicate_api_token
        )
        
        self.openai = self._init_api(
            OpenAIWrapper, 
            "OpenAI",
            api_key=openai_api_key
        )
        # Initialize News API client
        self.news_client = self._init_api(News, "News Aggregator")
        
        # Check email credentials
        self.email_available = bool(os.getenv("EMAIL_USER") and os.getenv("EMAIL_PASS"))
        if not self.email_available:
            logger.warning("Email credentials not found in environment variables (EMAIL_USER, EMAIL_PASS).")
    
    def _init_api(self, api_class, api_name, **kwargs):
        """Helper to initialize API clients with error handling and optional API keys."""
        try:
            # Pass specific keys if provided, otherwise rely on class defaults (env vars)
            client = api_class(**kwargs)
            logger.info(f"{api_name} API initialized successfully.")
            return client
        except Exception as e:
            logger.warning(f"Failed to initialize {api_name} API: {e}")
            return None
    
    #################################
    # WEB SEARCH FUNCTIONS
    #################################
    
    def web_crawl(
        self,
        query: str,
        sources: Union[Literal["all"], Literal["ddg"], Literal["wiki"], Literal["exa"], List[str]] = "all",
        num_results: int = 5,
        include_wiki_content: bool = False,
        max_wiki_sentences: int = 5,
        safe_search: str = "moderate",
        exa_max_chars: int = 2000
    ) -> Dict[str, Any]:
        """
        Perform comprehensive web research using selected sources (DuckDuckGo, Wikipedia, Exa).
        
        Args:
            query: Research query string
            sources: Sources to use - "all", "ddg", "wiki", "exa", or a list ["ddg", "wiki", "exa"]
            num_results: Maximum results per source
            include_wiki_content: Whether to include full Wikipedia article content
            max_wiki_sentences: Maximum sentences in Wikipedia summary (if not full content)
            safe_search: Safety level for DuckDuckGo search ('on', 'moderate', 'off')
            exa_max_chars: Maximum characters to return per Exa result
            
        Returns:
            Dictionary with search results from requested sources and a merged context string.
        """
        if self.web_crawler is None:
            logger.error("Web crawler not initialized.")
            return {"error": "Web crawler not available"}
        
        try:
            logger.info(f"Performing web crawl for query='{query}' sources='{sources}' num_results={num_results}")
            results = self.web_crawler.search_web(
                query=query,
                sources=sources,
                num_results=num_results,
                include_wiki_content=include_wiki_content,
                max_wiki_sentences=max_wiki_sentences,
                safe_search=safe_search,
                exa_max_chars=exa_max_chars
            )
            
            # The search_web method already returns the desired dictionary format
            # including the merged context.
            return results
        
        except Exception as e:
            logger.error(f"Error during web crawl for query='{query}': {e}", exc_info=True)
            return {"error": f"Web crawl failed: {str(e)}"}
    
    #################################
    # NEWS FUNCTIONS
    #################################
    
    def get_news(
        self,
        query: str,
        max_results: int = 10,
        from_date: Optional[Union[str, datetime]] = None,
        to_date: Optional[Union[str, datetime]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetches and combines news articles from available sources based on a query.
        
        Args:
            query: The search keyword or phrase.
            max_results: Approximate total number of articles desired (default: 10).
            from_date: Optional start date (YYYY-MM-DD or datetime object).
            to_date: Optional end date (YYYY-MM-DD or datetime object).
            
        Returns:
            List of normalized news articles, sorted by published date (newest first),
            or a list containing an error dictionary if the client is unavailable or fails.
        """
        if self.news_client is None:
            logger.error("News API client not initialized.")
            return [{"error": "News API client not available"}]
        
        try:
            logger.info(f"Fetching news with query='{query}', max_results={max_results}")
            news_articles = self.news_client.get_news(
                query=query,
                max_results=max_results,
                from_date=from_date,
                to_date=to_date
            )
            logger.info(f"Found {len(news_articles)} news articles for query='{query}'")
            return news_articles
        
        except Exception as e:
            logger.error(f"Error fetching news for query='{query}': {e}", exc_info=True)
            # Return the error message in the expected list format
            return [{"error": f"Failed to fetch news: {str(e)}"}]
    
    #################################
    # MEDIA GENERATION FUNCTIONS
    #################################
    
    def generate_image(
        self,
        prompt: str,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Generate an image from a text prompt using Replicate's model.
        
        Args:
            prompt: Text description of the desired image
            save_path: Optional path to save the image (will auto-generate if None)

        
        Returns:
            str: Either:
            - Path to the saved image (if save successful)
            - URL to the generated image (if save failed)
            - Error message string if generation failed
        """
        if self.replicate is None:
            logger.error("Replicate API not available for image generation.")
            return "Error: Replicate API not available"
        
        try:
            logger.info(f"Generating image with prompt: '{prompt[:50]}...'")
            response = self.replicate.generate_image(
                prompt=prompt,
                aspect_ratio="1:1",  # Default square aspect ratio
                model="flux-dev"
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
                # Ensure data/output/test_images exists relative to project root
                save_dir = os.path.join(project_root, "data", "output", "test_images")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, filename)
            else:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Download image
            try:
                from urllib.request import urlretrieve
                urlretrieve(image_url, save_path)
                logger.info(f"Image saved to: {save_path}")
                return image_url
            except Exception as e:
                logger.warning(f"Error saving image to {save_path}: {e}")
                return image_url
            
        except Exception as e:
            logger.error(f"Error generating image: {e}", exc_info=True)
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
            logger.error("Replicate API not available for music generation.")
            return "Error: Replicate API not available"
        
        try:
            logger.info(f"Generating music with prompt: '{prompt[:50]}...' duration: {duration}s")
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
                    # Ensure save dir exists relative to project root
                    save_dir = os.path.join(project_root, "data", "output", "music")
                    os.makedirs(save_dir, exist_ok=True)
                    output_file_path = os.path.join(save_dir, os.path.basename(save_path))
                    downloaded_path = self.replicate.download_file(music_url, output_dir=save_dir, filename=os.path.basename(save_path))
                    logger.info(f"Music saved to: {downloaded_path}")
                    return music_url
                except Exception as e:
                    logger.warning(f"Could not download music file: {e}")
                    return music_url
            
            return music_url
            
        except Exception as e:
            logger.error(f"Error generating music: {e}", exc_info=True)
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
            logger.error("Replicate API not available for video generation.")
            return "Error: Replicate API not available"
        
        try:
            logger.info(f"Generating video from '{image_url}' with prompt: '{motion_prompt[:50]}...'")
            video_url = self.replicate.generate_video(
                image_url=image_url,
                prompt=motion_prompt,
                num_frames=81  # Minimum required by the model
            )
            
            if not video_url:
                return "Failed to generate video"
                
            # Download if save path provided
            if save_path:
                try:
                    if hasattr(self.replicate, 'download_file'):
                        # Ensure save dir exists relative to project root
                        save_dir = os.path.join(project_root, "data", "output", "videos")
                        os.makedirs(save_dir, exist_ok=True)
                        output_file_path = os.path.join(save_dir, os.path.basename(save_path))
                        downloaded_path = self.replicate.download_file(video_url, output_dir=save_dir, filename=os.path.basename(save_path))
                        logger.info(f"Video saved to: {downloaded_path}")
                    else:
                        logger.warning("download_file method not found on ReplicateAPI instance")
                except Exception as e:
                    logger.warning(f"Could not download video file: {e}")
                return video_url
            
            return video_url
        except Exception as e:
            logger.error(f"Error generating video: {e}", exc_info=True)
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
            logger.error("Replicate API not available for music video generation.")
            return "Error: Replicate API not available"
        
        try:
            logger.info(f"Music video step 1: Generating image for '{prompt[:30]}...'")
            image_url = self.generate_image(prompt)
            if not image_url or image_url.startswith("Error") or image_url.startswith("Failed"):
                return f"Failed at image generation step: {image_url}"
                
            # Create slightly modified prompts for variety
            video_prompt = f"Camera slowly exploring {prompt}, with smooth movement"
            music_prompt = f"Soundtrack for {prompt}, emotionally fitting the visual scene"
            
            logger.info("Music video step 2: Generating video...")
            video_url = self.generate_video(image_url, video_prompt)
            if not video_url or video_url.startswith("Error") or video_url.startswith("Failed"):
                return f"Failed at video generation step: {video_url}"
                
            logger.info("Music video step 3: Generating music...")
            music_url = self.generate_music(music_prompt, duration=5)
            if not music_url or music_url.startswith("Error") or music_url.startswith("Failed"):
                return f"Failed at music generation step: {music_url}"
            
            # Download files
            if not hasattr(self.replicate, 'download_file') or not hasattr(self.replicate, 'merge_video_audio'):
                logger.error("Required methods (download_file, merge_video_audio) not found.")
                return "Error: Required methods not found."
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = save_path or 'music_video'
            # Define paths within project structure
            video_dir = os.path.join(project_root, "data", "output", "videos")
            music_dir = os.path.join(project_root, "data", "output", "music")
            merged_dir = os.path.join(project_root, "data", "output", "videos") # Save merged in videos
            
            os.makedirs(video_dir, exist_ok=True)
            os.makedirs(music_dir, exist_ok=True)
            os.makedirs(merged_dir, exist_ok=True)

            logger.info("Music video step 3b: Downloading video and audio...")
            video_filename = f"{base_name}_video_{timestamp}.mp4"
            music_filename = f"{base_name}_audio_{timestamp}.mp3"
            merged_filename = f"{base_name}_merged_{timestamp}.mp4"
            
            video_path = self.replicate.download_file(video_url, output_dir=video_dir, filename=video_filename)
            music_path = self.replicate.download_file(music_url, output_dir=music_dir, filename=music_filename)
            
            if not video_path or not music_path:
                logger.error(f"Failed to download files. Video URL: {video_url}, Music URL: {music_url}")
                return f"Failed to download files. Video: {video_url}, Music: {music_url}"
            
            # Merge video and audio
            logger.info("Music video step 4: Merging video and audio...")
            merged_path = self.replicate.merge_video_audio(video_path, music_path, output_dir=merged_dir, filename=merged_filename)
            
            if merged_path:
                logger.info(f"Music video merged successfully: {merged_path}")
                return merged_path
            else:
                logger.error("Music video merge failed.")
                return f"Video: {video_path}, Music: {music_path} (Merge failed)"
            
        except Exception as e:
            logger.error(f"Error generating music video: {e}", exc_info=True)
            return f"Music video generation failed: {str(e)}"
    
    def generate_threed(
        self,
        image_url: str,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Generate 3D model from an image using Replicate's Trellis model.
        Saves the model locally if save_path is provided, but returns the model URL.
        
        Args:
            image_url: URL or local path to the source image.
            # prompt: Optional textual description (currently unused by Trellis).
            save_path: Optional path to save the 3D model (.glb).
            # remove_background: Whether to attempt background removal from the source image.
            # seed: Optional random seed for reproducibility.
            
        Returns:
            URL of the generated 3D model, or an error message string.
        """
        if not self.replicate:
            logger.error("Replicate API not available for 3D generation.")
            return "Error: Replicate API not available"
            
        if not image_url:
            logger.error("Image URL or path is required for 3D generation.")
            return "Error: Image URL or path is required."
        
        try:
            # --- Prepare Image Input ---
            # Check if image_url is a local path and needs uploading
            prepared_image_url = image_url # Assume it's a URL initially
            if os.path.exists(image_url):
                logger.info(f"Local image path detected: {image_url}. Preparing...")
                # Use the upload functionality within replicate_API if available
                if hasattr(self.replicate, 'prepare_image_input'):
                     prepared_image_url = self.replicate.prepare_image_input(image_url)
                     if not prepared_image_url or not isinstance(prepared_image_url, str):
                         logger.error(f"Failed to prepare/upload local image: {image_url}")
                         return f"Error: Failed to prepare/upload local image: {image_url}"
                     logger.info(f"Image prepared: {prepared_image_url}")
                else:
                    logger.error("ReplicateAPI instance missing prepare_image_input method.")
                    return "Error: ReplicateAPI instance missing prepare_image_input method."
            elif not isinstance(image_url, str) or not image_url.startswith(('http', 'https')):
                 logger.error(f"Invalid image_url provided: {image_url}")
                 return f"Error: Invalid image_url provided: {image_url}"
            
            # --- Generate 3D Model URL ---
            logger.info(f"Generating 3D model from image: {prepared_image_url}...")
            threed_url = self.replicate.generate_threed(
                image_url=prepared_image_url,
                model="trellis"
                )
            
            if not threed_url:
                raise Exception("Replicate returned empty result for 3D generation")

            # Ensure threed_url is a string URL
            if not isinstance(threed_url, str):
                 # Handle cases where the output might be a dict or FileOutput object
                 if hasattr(threed_url, 'url'):
                     threed_url = threed_url.url
                 elif isinstance(threed_url, dict) and 'mesh' in threed_url:
                     mesh_output = threed_url['mesh']
                     threed_url = mesh_output.url if hasattr(mesh_output, 'url') else str(mesh_output)
                 else:
                     threed_url = str(threed_url) # Fallback conversion
            
            if not threed_url or not threed_url.startswith('http'):
                raise ValueError(f"Invalid URL: {threed_url}")

            # --- Download Locally (if requested) ---
            if save_path:
                # Ensure save dir exists relative to project root
                save_dir = os.path.join(project_root, "data", "output", "threed_models")
                os.makedirs(save_dir, exist_ok=True)
                output_file_path = os.path.join(save_dir, os.path.basename(save_path))
                logger.info(f"Downloading 3D model to: {output_file_path}")
                if hasattr(self.replicate, 'download_file'):
                    downloaded_path = self.replicate.download_file(threed_url, output_dir=save_dir, filename=os.path.basename(save_path))
                    if downloaded_path:
                        logger.info(f"3D Model saved locally to: {downloaded_path}")
                    else:
                        logger.warning(f"Failed to download 3D model locally to {output_file_path}")
                else:
                    logger.warning("download_file method not available. Cannot save locally.")
            
            # --- Return the URL --- 
            return threed_url
            
        except Exception as e:
            logger.error(f"Error generating 3D model: {e}", exc_info=True)
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
            logger.info(f"Opening URL in browser: {url}")
            webbrowser.open(url)
            return True
        except Exception as e:
            logger.warning(f"Error opening URL {url}: {e}")
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
            logger.warning("Attempted send_email, but credentials not configured.")
            return False
        
        try:
            # Get credentials from environment
            email_user = sender or os.getenv("EMAIL_USER")
            email_pass = os.getenv("EMAIL_PASS")
            
            if not email_user or not email_pass:
                logger.error("Email credentials missing even after check.")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg["From"] = email_user
            msg["To"] = recipient
            msg["Subject"] = subject
            
            # Attach body with appropriate type
            content_type = "html" if html else "plain"
            msg.attach(MIMEText(body, content_type))
            
            # Setup and send email
            logger.info(f"Sending email to {recipient} with subject: {subject}")
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(email_user, email_pass)
            server.send_message(msg)
            server.quit()
            logger.info(f"Email sent successfully to {recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email to {recipient}: {e}", exc_info=True)
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
            # Import from I_integrations package
            from I_integrations.openweather_API import OpenWeatherAPI
            
            # Initialize API if needed
            if not hasattr(self, "weather_api"):
                self.weather_api = OpenWeatherAPI()
            
            # Get current weather
            logger.info(f"Getting current weather for {location} (units: {units})")
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
                    "description": weather_data["weather"][0].get("description", ""),
                    "condition": weather_data["weather"][0].get("main", ""),
                    "icon": weather_data["weather"][0].get("icon", ""),
                    "units": units,
                    "timestamp": weather_data.get("dt"),
                    "timezone": weather_data.get("timezone"),
                    # "raw_data": weather_data
                }
                return formatted_result
            else:
                logger.warning(f"Weather data format unexpected for {location}: {weather_data}")
                return {"error": "Weather data not available", "raw_data": weather_data}
            
        except ModuleNotFoundError:
            logger.error("OpenWeatherAPI module not found in I_integrations.")
            return {"error": "Weather functionality unavailable due to missing module."}
        except Exception as e:
            logger.error(f"Error getting weather for {location}: {e}", exc_info=True)
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
            # Import from I_integrations package
            from I_integrations.openweather_API import OpenWeatherAPI
            
            # Initialize API if needed
            if not hasattr(self, "weather_api"):
                self.weather_api = OpenWeatherAPI()
            
            # Get forecast
            logger.info(f"Getting {days}-day forecast for {location} (units: {units})")
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
                            "description": entry.get("weather", [{}])[0].get("description", ""),
                            "condition": entry.get("weather", [{}])[0].get("main", ""),
                            "icon": entry.get("weather", [{}])[0].get("icon", ""),
                            "wind_speed": entry.get("wind", {}).get("speed"),
                        }
                        
                        days_forecast[date_str].append(forecast_entry)
                
                return {
                    "location": city_info.get("name", location),
                    "country": city_info.get("country", ""),
                    "timezone": city_info.get("timezone"),
                    "days": days_forecast,
                    "units": units,
                    # "raw_data": forecast_data
                }
            else:
                logger.warning(f"Forecast data format unexpected for {location}: {forecast_data}")
                return {"error": "Forecast data not available", "raw_data": forecast_data}
            
        except ModuleNotFoundError:
            logger.error("OpenWeatherAPI module not found in I_integrations.")
            return {"error": "Weather functionality unavailable due to missing module."}
        except Exception as e:
            logger.error(f"Error getting forecast for {location}: {e}", exc_info=True)
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
            logger.error("OpenAI API not available for text-to-speech.")
            return "Error: OpenAI API not available"
        
        # Validate voice parameter against OpenAI's allowed voices
        allowed_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "ash", "sage", "coral"]
        if voice not in allowed_voices:
            logger.warning(f"'{voice}' is not a valid OpenAI voice. Using 'alloy' instead.")
            voice = "alloy"
        
        try:
            # Call OpenAI's text-to-speech API
            logger.info(f"Generating speech for text: '{text[:50]}...' voice: {voice} speed: {speed}")
            # Determine save path relative to project root
            if not output_path:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"speech_{timestamp}.mp3"
                save_dir = os.path.join(project_root, "data", "output", "audio")
                os.makedirs(save_dir, exist_ok=True)
                output_path = os.path.join(save_dir, filename)
            else:
                # Ensure directory exists if a full path is provided
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
            audio_path = self.openai.text_to_speech(
                text=text,
                voice=voice,
                output_path=output_path,
                speed=speed
            )
            logger.info(f"Text-to-speech generated and saved to: {audio_path}")
            return audio_path
        except Exception as e:
            logger.error(f"Error generating speech: {e}", exc_info=True)
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
            logger.error("OpenAI API not available for speech transcription.")
            return {"error": "OpenAI API not available"}
        
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found for transcription: {audio_file}")
            return {"error": f"Audio file not found: {audio_file}"}
        
        try:
            # Call OpenAI's transcription API
            logger.info(f"Transcribing audio file: {audio_file} Language: {language or 'auto'}")
            result = self.openai.transcribe_audio(
                audio_file=audio_file,
                language=language,
                prompt=prompt
            )
            logger.info(f"Transcription successful for {audio_file}. Text length: {len(result.get('text', ''))}")
            return result
        except Exception as e:
            logger.error(f"Error transcribing speech for file {audio_file}: {e}", exc_info=True)
            return {"error": f"Transcription failed: {str(e)}"}


# Example usage
if __name__ == "__main__":
    # Configure root logger for demo purposes if needed
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("\n===== TOOLS DEMO (using logging) =====\n")
    tools = Tools()
    
    # Function to prompt for input with default value
    def prompt(message, default=None):
        result = input(f"{message} [{default}]: ") if default else input(f"{message}: ")
        return result if result.strip() else default
    
    # Demo menu
    while True:
        print("\nAvailable Tools:")
        print("1. Web Search")
        print("2. News Search")
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
            sources_options = "all (All Sources), ddg (DuckDuckGo), wiki (Wikipedia), exa (Exa)"
            sources = prompt(f"Select sources [{sources_options}]", "all")
            num_results = int(prompt("Number of results per source", "5"))
            include_wiki_content = prompt("Include full Wikipedia content? (y/n)", "n").lower() == "y"
            
            print(f"\nSearching the web for: '{query}' using {sources}")
            results = tools.web_crawl(
                query=query, 
                sources=sources,
                num_results=num_results,
                include_wiki_content=include_wiki_content
            )
            
            # Display search results
            print(f"\nSources used: {', '.join(results.get('sources_used', []))}")
            
            # Display Wikipedia results if available
            if "wiki_results" in results:
                wiki_data = results["wiki_results"]
                print(f"\nWikipedia: {wiki_data.get('title', '')}")
                print(f"Summary: {wiki_data.get('summary', '')[:300]}...")
                if include_wiki_content and "content" in wiki_data:
                    print(f"Content preview: {wiki_data.get('content', '')[:300]}...")
            
            # Display DuckDuckGo results if available
            if "ddg_results" in results:
                print("\nDuckDuckGo Results:")
                for i, result in enumerate(results["ddg_results"][:3], 1):
                    print(f"\n{i}. {result.get('title', '')}")
                    print(f"   URL: {result.get('link', '')}")
                    print(f"   {result.get('snippet', '')[:100]}...")
            
            # Display Exa results if available
            if "exa_results" in results:
                print("\nExa Results:")
                for i, result in enumerate(results["exa_results"][:3], 1):
                    print(f"\n{i}. {result.get('title', '')}")
                    print(f"   URL: {result.get('url', '')}")
                    if result.get('text'):
                        print(f"   {result.get('text', '')[:100]}...")
                
        elif choice == "2":
            query = prompt("Enter news search query", "latest technology")
            max_results = int(prompt("Maximum number of articles", "5"))
            from_date = prompt("From date (YYYY-MM-DD, optional)", "")
            to_date = prompt("To date (YYYY-MM-DD, optional)", "")
            
            print(f"\nSearching news for: {query}...")
            news_results = tools.get_news(
                query=query, 
                max_results=max_results,
                from_date=from_date if from_date else None,
                to_date=to_date if to_date else None
            )
            
            if news_results and "error" not in news_results[0]:
                print(f"\nFound {len(news_results)} news articles:")
                for i, article in enumerate(news_results, 1):
                    print(f"\n{i}. {article.get('title', 'No title')}")
                    print(f"   Source: {article.get('source', 'Unknown')}")
                    print(f"   Date: {article.get('published_at', 'Unknown date')}")
                    print(f"   URL: {article.get('url', 'No URL')}")
                    if article.get('description'):
                        print(f"   {article.get('description', '')[:150]}...")
            else:
                error_msg = news_results[0].get('error', 'Unknown error') if news_results else "No results"
                print(f"Error: {error_msg}")
                
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
            motion_prompt = prompt("Enter motion description", "Camera slowly panning around the scene, revealing details")
            if not image_url:
                print("Image URL is required for video generation.")
                continue
            print("\nGenerating video, please wait...")
            result = tools.generate_video(image_url, motion_prompt)
            print(f"Result: {result}")
            
        elif choice == "6":
            prompt_text = prompt("Enter music video concept", "A cosmic journey through nebulae and star formations")
            print("\nGenerating complete music video (this may take several minutes)...")
            result = tools.generate_music_video(prompt_text)
            print(f"Result: {result}")
            
        elif choice == "7":
            # Now uses Replicate Trellis (Image-to-3D only)
            image_url = prompt("Enter image URL or local path", "")
            if not image_url:
                print("Image URL or path is required for 3D generation.")
                continue
                
            save_path_input = prompt("Enter local save path (optional, e.g., output/threed/my_model.glb)", "")
                
            print("\nGenerating 3D model from image using Replicate Trellis, please wait...")
            # Call the simplified tools.py function
            result = tools.generate_threed(
                image_url=image_url,
                save_path=save_path_input if save_path_input else None
            )
                
            print(f"Result (URL): {result}")
            if save_path_input:
                 print(f"(Attempted to save locally to {save_path_input})")
            
        elif choice == "8":
            format_type = prompt("Format (iso/human/date/time)", "human")
            result = tools.get_current_datetime(format_type)
            print(f"Current datetime: {result}")
            
        elif choice == "9":
            url = prompt("Enter URL to open", "https://www.google.com")
            result = tools.open_url_in_browser(url)
            if result:
                print(f"Successfully opened {url} in browser.")
            else:
                print(f"Failed to open {url} in browser.")
            
        elif choice == "10":
            if not tools.email_available:
                print("Email is not configured. Add EMAIL_USER and EMAIL_PASS to .env file.")
                continue
                
            recipient = prompt("Enter recipient email")
            subject = prompt("Enter subject", "Test email from Tools")
            body = prompt("Enter message", "This is a test email sent from the Tools module.")
            html_format = prompt("Send as HTML? (y/n)", "n").lower() == "y"
            
            if not recipient:
                print("Recipient email is required.")
                continue
                
            print("\nSending email...")
            result = tools.send_email(recipient, subject, body, html=html_format)
            if result:
                print("Email sent successfully!")
            else:
                print("Failed to send email.")
                
        elif choice == "11":
            location = prompt("Enter location (e.g., 'London,UK', 'New York,US')", "London,UK")
            units = prompt("Units (metric/imperial)", "metric")
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
            units = prompt("Units (metric/imperial)", "metric")
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
                
        elif choice == "13":
            text = prompt("Enter text to convert to speech", "Welcome to the demo of our text to speech capability.")
            voice = prompt("Choose voice (alloy, echo, fable, onyx, nova, shimmer, ash, sage, coral)", "alloy")
            output_path = prompt("Enter output file name (optional)", "demo_speech.mp3")
            speed = float(prompt("Speech speed (0.25 to 4.0)", "1.0"))
            
            print("\nGenerating speech...")
            result = tools.text_to_speech(text, voice, output_path, speed=speed)
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
    
    logger.info("\n===== DEMO COMPLETE ======") 