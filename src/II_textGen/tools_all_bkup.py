import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
import inspect
from functools import wraps
from datetime import datetime, timedelta
import time

# Import utilities using FileFinder
from modules.VIII_utils.file_finder import FileFinder
finder = FileFinder()

# Import LocalSystem API and Schema Manager
LocalSystem = finder.get_class('localSystem_API.py', 'LocalSystem')
ToolsSchemas = finder.get_class('tools_schemas.py', 'ToolsSchemas')

# Import base LLM components
Provider = finder.get_class('base_LLM.py', 'Provider')

# Add new imports
ImageGen = finder.get_class('imageGen.py', 'ImageGen')
ThreedGen = finder.get_class('threedGen.py', 'ThreedGen')
MusicGen = finder.get_class('musicGen.py', 'MusicGen')
GmailWrapper = finder.get_class('gmail_API.py', 'GmailWrapper')
GSearchAPI = finder.get_class('gsearch_API.py', 'GSearchAPI')
GMapsAPI = finder.get_class('gmaps_API.py', 'GMapsAPI')
NewsAPI = finder.get_class('newsapi_API.py', 'NewsAPI')
GNewsAPI = finder.get_class('gnews_API.py', 'GNewsAPI')
OpenWeatherAPI = finder.get_class('openweather_API.py', 'OpenWeatherAPI')
TomorrowAPI = finder.get_class('tomorrow_API.py', 'TomorrowAPI')
DDGWrapper = finder.get_class('DDG_API.py', 'DDGWrapper')
ExaWrapper = finder.get_class('Exa_API.py', 'ExaWrapper')
WikiWrapper = finder.get_class('Wiki_API.py', 'WikiWrapper')
TwilioAPI = finder.get_class('twilio_API.py', 'TwilioAPI')

class Tools:
    """Core tools for ArX system."""
    
    def __init__(self):
        """Initialize Tools with all integrations."""
        # Initialize existing integrations
        self.local_system = LocalSystem()
        
        # Initialize generation systems
        self.image_gen = ImageGen()
        self.threed_gen = ThreedGen()
        self.music_gen = MusicGen()
        
        # Initialize Google APIs
        self.gmail = GmailWrapper()
        self.gsearch = GSearchAPI()
        self.gmaps = GMapsAPI()
        
        # Initialize schema manager with OpenAI
        self.schema_manager = ToolsSchemas(provider=Provider.OPENAI)
        
        # Initialize News APIs
        self.newsapi = NewsAPI()
        self.gnews = GNewsAPI()
        
        # Initialize Weather APIs
        self.openweather = OpenWeatherAPI()
        self.tomorrow = TomorrowAPI()
        
        # Initialize Web Crawling APIs
        self.ddg = DDGWrapper()
        self.exa = ExaWrapper()
        self.wiki = WikiWrapper()
        
        # Initialize WhatsApp/Twilio API
        self.twilio = TwilioAPI()
        
        # Register all tools
        self.schema_manager.register_tools_from_class(self)
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict]:
        """Get the schema for a specific tool."""
        return self.schema_manager.get_schema(tool_name)
    
    # Tool Methods
    def capture_webcam(self, save_path: Optional[str] = None, show_preview: bool = True) -> str:
        """
        Capture an image from the webcam.
        
        :param save_path: Optional path to save the captured image
        :param show_preview: Whether to show preview window during capture
        :return: Path to saved image or error message
        """
        result = self.local_system.capture_webcam(save_path, show_preview)
        return str(result) if result else "Failed to capture webcam image"
    
    def capture_screenshot(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Capture a screenshot of the screen or specified region.
        
        :param region: Optional tuple of (left, top, width, height) for region capture
        :param save_path: Optional path to save the screenshot
        :return: Path to saved screenshot or error message
        """
        result = self.local_system.capture_screenshot(region, save_path)
        return str(result) if result else "Failed to capture screenshot"
    
    def plot_data(
        self,
        x_data: List[float],
        y_data: List[float],
        title: str = "Data Visualization",
        xlabel: str = "X",
        ylabel: str = "Y",
        save_path: Optional[str] = None
    ) -> str:
        """
        Create and save a plot using provided data.
        
        :param x_data: List of x-axis values
        :param y_data: List of y-axis values
        :param title: Plot title
        :param xlabel: X-axis label
        :param ylabel: Y-axis label
        :param save_path: Optional path to save the plot
        :return: Path to saved plot or error message
        """
        import numpy as np
        result = self.local_system.plot_data(
            np.array(x_data),
            np.array(y_data),
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            save_path=save_path
        )
        return str(result) if result else "Failed to create plot"
    
    def record_webcam_video(
        self,
        duration: float = 10.0,
        fps: int = 30,
        save_path: Optional[str] = None,
        show_preview: bool = True
    ) -> str:
        """
        Record video from webcam.
        
        :param duration: Recording duration in seconds
        :param fps: Frames per second
        :param save_path: Optional path to save video
        :param show_preview: Whether to show preview during recording
        :return: Path to saved video or error message
        """
        result = self.local_system.capture_webcam_video(
            duration=duration,
            fps=fps,
            save_path=save_path,
            show_preview=show_preview
        )
        return str(result) if result else "Failed to record webcam video"
    
    def record_screen(
        self,
        duration: float = 10.0,
        fps: int = 30,
        region: Optional[Tuple[int, int, int, int]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Record screen activity.
        
        :param duration: Recording duration in seconds
        :param fps: Frames per second
        :param region: Optional tuple of (left, top, width, height) for region capture
        :param save_path: Optional path to save recording
        :return: Path to saved recording or error message
        """
        result = self.local_system.capture_screen_video(
            duration=duration,
            fps=fps,
            region=region,
            save_path=save_path
        )
        return str(result) if result else "Failed to record screen"
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information including OS, CPU, memory usage, etc.
        
        :return: Dictionary containing system information
        """
        return self.local_system.get_system_info()

    async def generate_image(
        self,
        prompt: str,
        model: str = "flux",
        negative_prompt: str = "",
        size: str = "square_hd",
        save_path: Optional[str] = None,
        preview: bool = True,
        use_magic_prompt: bool = True
    ) -> str:
        """
        Generate an image using various AI models.
        
        :param prompt: Text description of the image to generate
        :param model: Model to use ('flux', 'recraft', 'sd3l', 'sdu')
        :param negative_prompt: Things to avoid in the generation
        :param size: Image size/aspect ratio
        :param save_path: Optional path to save the image
        :param preview: Whether to show preview
        :param use_magic_prompt: Whether to enhance prompt with LLM
        :return: Path to generated image or error message
        """
        try:
            method_name = f"generate_{model.lower()}"
            if not hasattr(self.image_gen, method_name):
                return f"Unknown model: {model}"
            
            generate_method = getattr(self.image_gen, method_name)
            result = await generate_method(
                prompt=prompt,
                negative_prompt=negative_prompt,
                size=size,
                save_path=save_path,
                preview=preview,
                use_magic_prompt=use_magic_prompt
            )
            return str(result) if result else f"Failed to generate image with {model}"
            
        except Exception as e:
            return f"Image generation error: {str(e)}"

    async def generate_3d(
        self,
        prompt: str,
        provider: str = "meshy",
        image_path: Optional[str] = None,
        preview: bool = True,
        use_magic_prompt: bool = True,
        refine_meshy: bool = False
    ) -> str:
        """
        Generate a 3D model from text or image.
        
        :param prompt: Text description of the 3D model
        :param provider: Provider to use ('meshy', 'tripo', 'stability', 'trellis')
        :param image_path: Optional path to input image
        :param preview: Whether to preview the result
        :param use_magic_prompt: Whether to enhance text prompt
        :param refine_meshy: Whether to refine Meshy output for finished results or you just want to show a low resolution preview
        :return: Path to generated 3D model or error message
        """
        try:
            result = await self.threed_gen.generate(
                prompt=prompt,
                image_path=image_path,
                provider=provider,
                preview=preview,
                use_magic_prompt=use_magic_prompt,
                refine_meshy=refine_meshy
            )
            return str(result) if result else f"Failed to generate 3D model with {provider}"
            
        except Exception as e:
            return f"3D generation error: {str(e)}"

    async def generate_music(
        self,
        prompt: str,
        model: str = "meta",
        duration: int = 8,
        save_path: Optional[str] = None,
        use_magic_prompt: bool = True,
        play_preview: bool = True
    ) -> str:
        """
        Generate music from text description.
        
        :param prompt: Text description of the music
        :param model: Model to use ('meta', 'stable')
        :param duration: Length in seconds
        :param save_path: Optional path to save audio
        :param use_magic_prompt: Whether to enhance prompt
        :param play_preview: Whether to play the generated audio
        :return: Path to generated audio or error message
        """
        try:
            method_name = f"generate_{model.lower()}"
            if not hasattr(self.music_gen, method_name):
                return f"Unknown model: {model}"
            
            generate_method = getattr(self.music_gen, method_name)
            result = await generate_method(
                prompt=prompt,
                duration=duration,
                save_path=save_path,
                use_magic_prompt=use_magic_prompt
            )
            
            if result and play_preview:
                await self.music_gen.play_audio(result)
                
            return str(result) if result else f"Failed to generate music with {model}"
            
        except Exception as e:
            return f"Music generation error: {str(e)}"

    # Gmail Core Functions
    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        attachments: Optional[List[str]] = None
    ) -> str:
        """
        Send an email using Gmail.
        
        :param to: Recipient email address
        :param subject: Email subject
        :param body: Email body content
        :param attachments: Optional list of file paths to attach
        :return: Success message or error
        """
        try:
            result = self.gmail.send_email(to, subject, body, attachments)
            return "Email sent successfully" if result else "Failed to send email"
        except Exception as e:
            return f"Email error: {str(e)}"

    def get_recent_emails(self, max_results: int = 5) -> List[Dict]:
        """
        Get recent emails from Gmail inbox.
        
        :param max_results: Maximum number of emails to retrieve
        :return: List of email details
        """
        try:
            return self.gmail.list_messages(max_results=max_results)
        except Exception as e:
            return [{"error": f"Failed to get emails: {str(e)}"}]

    # Google Search Core Functions
    def web_search(
        self,
        query: str,
        num_results: int = 5,
        language: str = "en"
    ) -> List[Dict]:
        """
        Perform a web search.
        
        :param query: Search query
        :param num_results: Number of results to return
        :param language: Search language
        :return: List of search results
        """
        try:
            results = self.gsearch.search(
                query=query,
                num=num_results,
                language=language
            )
            return results.get('items', [])
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]

    def image_search(
        self,
        query: str,
        num_results: int = 5,
        size: str = "large"
    ) -> List[Dict]:
        """
        Search for images.
        
        :param query: Search query
        :param num_results: Number of results
        :param size: Image size preference
        :return: List of image results
        """
        try:
            results = self.gsearch.search_images(
                query=query,
                num=num_results,
                size=size.upper()
            )
            return results.get('items', [])
        except Exception as e:
            return [{"error": f"Image search failed: {str(e)}"}]

    def news_search(
        self,
        query: str,
        num_results: int = 5,
        recent: bool = True
    ) -> List[Dict]:
        """
        Search for news articles.
        
        :param query: Search query
        :param num_results: Number of results
        :param recent: Whether to prioritize recent news
        :return: List of news articles
        """
        try:
            results = self.gsearch.search_news(
                query=query,
                num=num_results,
                recent=recent
            )
            return results.get('items', [])
        except Exception as e:
            return [{"error": f"News search failed: {str(e)}"}]

    # Google Maps Core Functions
    def find_place(
        self,
        query: str,
        location: Optional[Tuple[float, float]] = None,
        radius: Optional[int] = None
    ) -> Dict:
        """
        Find a place using Google Maps.
        
        :param query: Search query
        :param location: Optional (latitude, longitude) tuple
        :param radius: Search radius in meters
        :return: Place details
        """
        try:
            results = self.gmaps.text_search(
                query=query,
                location=location,
                radius=radius
            )
            return results.get('results', [{}])[0]
        except Exception as e:
            return {"error": f"Place search failed: {str(e)}"}

    def get_directions(
        self,
        origin: str,
        destination: str,
        mode: str = "driving",
        alternatives: bool = True
    ) -> List[Dict]:
        """
        Get directions between two locations.
        
        :param origin: Starting location
        :param destination: Ending location
        :param mode: Travel mode (driving, walking, transit, bicycling)
        :param alternatives: Whether to return alternative routes
        :return: List of routes
        """
        try:
            return self.gmaps.get_directions(
                origin=origin,
                destination=destination,
                mode=mode,
                alternatives=alternatives
            )
        except Exception as e:
            return [{"error": f"Directions failed: {str(e)}"}]

    def get_place_details(
        self,
        place_id: str,
        fields: Optional[List[str]] = None
    ) -> Dict:
        """
        Get detailed information about a place.
        
        :param place_id: Google Maps place ID
        :param fields: List of fields to retrieve
        :return: Place details
        """
        try:
            return self.gmaps.place_details(
                place_id=place_id,
                fields=fields
            )
        except Exception as e:
            return {"error": f"Place details failed: {str(e)}"}

    # News Functions
    def get_news(
        self,
        query: str,
        provider: str = "newsapi",
        num_results: int = 5,
        language: str = "en",
        sort_by: str = "relevancy",
        include_content: bool = False
    ) -> List[Dict]:
        """
        Get news articles from specified provider.
        
        :param query: Search query
        :param provider: News provider ('newsapi' or 'gnews')
        :param num_results: Number of results to return
        :param language: Language code (e.g., 'en', 'fr')
        :param sort_by: Sort method ('relevancy', 'date')
        :param include_content: Whether to include full article content
        :return: List of news articles
        """
        try:
            if provider == "newsapi":
                results = self.newsapi.get_everything(
                    query=query,
                    language=language,
                    sort_by=sort_by,
                    page_size=num_results
                )
                return results.get('articles', [])
            
            elif provider == "gnews":
                results = self.gnews.search_news(
                    query=query,
                    lang=language,
                    max_results=num_results,
                    sort="relevance" if sort_by == "relevancy" else "publishedAt",
                    expand=include_content
                )
                return results.get('articles', [])
            
            else:
                return [{"error": f"Unknown provider: {provider}"}]
                
        except Exception as e:
            return [{"error": f"News search failed: {str(e)}"}]

    def get_top_headlines(
        self,
        category: str = "technology",
        country: str = "us",
        num_results: int = 5,
        provider: str = "newsapi"
    ) -> List[Dict]:
        """
        Get top headlines by category.
        
        :param category: News category (technology, business, etc.)
        :param country: Country code (e.g., 'us', 'gb')
        :param num_results: Number of results to return
        :param provider: News provider ('newsapi' or 'gnews')
        :return: List of news articles
        """
        try:
            if provider == "newsapi":
                results = self.newsapi.get_top_headlines(
                    category=category,
                    country=country,
                    page_size=num_results
                )
                return results.get('articles', [])
            
            elif provider == "gnews":
                results = self.gnews.get_top_headlines(
                    category=category,
                    country=country,
                    max_results=num_results
                )
                return results.get('articles', [])
            
            else:
                return [{"error": f"Unknown provider: {provider}"}]
                
        except Exception as e:
            return [{"error": f"Headlines fetch failed: {str(e)}"}]

    def search_tech_news(
        self,
        topic: str,
        days: int = 7,
        num_results: int = 5,
        providers: List[str] = ["newsapi", "gnews"]
    ) -> Dict[str, List[Dict]]:
        """
        Search technology news across multiple providers.
        
        :param topic: Tech topic to search for
        :param days: How many days back to search
        :param num_results: Number of results per provider
        :param providers: List of providers to use
        :return: Dict of results by provider
        """
        try:
            results = {}
            from_date = datetime.now() - timedelta(days=days)
            
            for provider in providers:
                if provider == "newsapi":
                    response = self.newsapi.get_everything(
                        query=f"{topic} technology",
                        language="en",
                        sort_by="relevancy",
                        from_date=from_date,
                        page_size=num_results
                    )
                    results["newsapi"] = response.get('articles', [])
                
                elif provider == "gnews":
                    response = self.gnews.search_news(
                        query=f"{topic} technology",
                        lang="en",
                        max_results=num_results,
                        from_date=from_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        sort="relevance"
                    )
                    results["gnews"] = response.get('articles', [])
            
            return results
            
        except Exception as e:
            return {"error": f"Tech news search failed: {str(e)}"}

    def get_news_sources(
        self,
        category: str = None,
        language: str = "en",
        country: str = "us"
    ) -> List[Dict]:
        """
        Get available news sources.
        
        :param category: Optional category filter
        :param language: Language code
        :param country: Country code
        :return: List of news sources
        """
        try:
            results = self.newsapi.get_sources(
                category=category,
                language=language,
                country=country
            )
            return results.get('sources', [])
        except Exception as e:
            return [{"error": f"Failed to get sources: {str(e)}"}]

    # Weather Functions
    def get_current_weather(
        self,
        location: Union[str, Tuple[float, float]],
        provider: str = "openweather",
        units: str = "metric"
    ) -> Dict:
        """
        Get current weather for a location.
        
        :param location: City name or (latitude, longitude) tuple
        :param provider: Weather provider ('openweather' or 'tomorrow')
        :param units: Units of measurement ('metric' or 'imperial')
        :return: Current weather data
        """
        try:
            if provider == "openweather":
                if isinstance(location, tuple):
                    location = f"{location[0]},{location[1]}"
                return self.openweather.get_current_weather(location, units)
            
            elif provider == "tomorrow":
                return self.tomorrow.get_realtime(location, units)
            
            else:
                return {"error": f"Unknown provider: {provider}"}
                
        except Exception as e:
            return {"error": f"Weather fetch failed: {str(e)}"}

    def get_weather_forecast(
        self,
        location: Union[str, Tuple[float, float]],
        provider: str = "openweather",
        days: int = 5,
        units: str = "metric",
        hourly: bool = False
    ) -> Dict:
        """
        Get weather forecast for a location.
        
        :param location: City name or (latitude, longitude) tuple
        :param provider: Weather provider ('openweather' or 'tomorrow')
        :param days: Number of days to forecast
        :param units: Units of measurement
        :param hourly: Whether to get hourly forecast
        :return: Weather forecast data
        """
        try:
            if provider == "openweather":
                if isinstance(location, tuple):
                    location = f"{location[0]},{location[1]}"
                return self.openweather.get_forecast(
                    location=location,
                    units=units,
                    days=days
                )
            
            elif provider == "tomorrow":
                timesteps = ["1h"] if hourly else ["1d"]
                end_time = datetime.now() + timedelta(days=days)
                return self.tomorrow.get_forecast(
                    location=location,
                    timesteps=timesteps,
                    units=units,
                    endTime=end_time
                )
            
            else:
                return {"error": f"Unknown provider: {provider}"}
                
        except Exception as e:
            return {"error": f"Forecast fetch failed: {str(e)}"}

    def get_weather_comparison(
        self,
        location: Union[str, Tuple[float, float]],
        units: str = "metric"
    ) -> Dict:
        """
        Compare current weather from both providers.
        
        :param location: City name or (latitude, longitude) tuple
        :param units: Units of measurement
        :return: Comparison of weather data
        """
        try:
            # Get data from both providers
            ow_data = self.get_current_weather(location, "openweather", units)
            tm_data = self.get_current_weather(location, "tomorrow", units)
            
            if "error" in ow_data or "error" in tm_data:
                return {"error": "One or both providers failed"}
            
            # Extract and compare key metrics
            comparison = {
                "openweather": {
                    "temperature": ow_data['main']['temp'],
                    "humidity": ow_data['main']['humidity'],
                    "wind_speed": ow_data['wind']['speed'],
                    "description": ow_data['weather'][0]['description']
                },
                "tomorrow": {
                    "temperature": tm_data['data']['values']['temperature'],
                    "humidity": tm_data['data']['values']['humidity'],
                    "wind_speed": tm_data['data']['values']['windSpeed']
                },
                "differences": {
                    "temperature": abs(ow_data['main']['temp'] - tm_data['data']['values']['temperature']),
                    "humidity": abs(ow_data['main']['humidity'] - tm_data['data']['values']['humidity']),
                    "wind_speed": abs(ow_data['wind']['speed'] - tm_data['data']['values']['windSpeed'])
                }
            }
            
            return comparison
            
        except Exception as e:
            return {"error": f"Weather comparison failed: {str(e)}"}

    def get_historical_weather(
        self,
        location: Union[str, Tuple[float, float]],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        units: str = "metric"
    ) -> Dict:
        """
        Get historical weather data (Tomorrow.io only).
        
        :param location: City name or (latitude, longitude) tuple
        :param start_time: Start time (default: 24h ago)
        :param end_time: End time (default: now)
        :param units: Units of measurement
        :return: Historical weather data
        """
        try:
            return self.tomorrow.get_historical(
                location=location,
                startTime=start_time,
                endTime=end_time,
                units=units
            )
        except Exception as e:
            return {"error": f"Historical weather fetch failed: {str(e)}"}

    # Web Search Functions
    def search_web(
        self,
        query: str,
        providers: List[str] = ["ddg", "exa"],
        num_results: int = 3,
        include_content: bool = False
    ) -> Dict[str, List[Dict]]:
        """
        Search the web using multiple providers.
        
        :param query: Search query
        :param providers: List of providers to use ('ddg', 'exa')
        :param num_results: Results per provider
        :param include_content: Whether to include full content (Exa only)
        :return: Results from each provider
        """
        results = {}
        
        try:
            if "ddg" in providers:
                ddg_results = self.ddg.search(
                    query=query,
                    num_results=num_results,
                    safesearch="moderate"
                )
                results["ddg"] = ddg_results

            if "exa" in providers:
                if include_content:
                    exa_results = self.exa.search_with_contents(
                        query=query,
                        num_results=num_results,
                        max_chars=1000,
                        highlights=True
                    )
                else:
                    exa_results = self.exa.search(
                        query=query,
                        num_results=num_results,
                        use_autoprompt=True
                    ).get('results', [])
                results["exa"] = exa_results

            return results
        except Exception as e:
            return {"error": f"Web search failed: {str(e)}"}

    def search_news_web(
        self,
        query: str,
        days_back: Optional[int] = 30,
        num_results: int = 3
    ) -> List[Dict]:
        """
        Search for news across providers.
        
        :param query: News search query
        :param days_back: How many days back to search
        :param num_results: Number of results per provider
        :return: Combined news results
        """
        try:
            # Get DuckDuckGo news
            ddg_news = self.ddg.search_news(
                query=query,
                num_results=num_results,
                timelimit="m" if days_back <= 30 else "y"
            )

            # Get Exa news with date filter
            from datetime import datetime, timedelta
            start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            exa_news = self.exa.search(
                query=query,
                num_results=num_results,
                start_published_date=start_date,
                category="news"
            ).get('results', [])

            # Combine and sort by date
            all_news = ddg_news + exa_news
            return sorted(all_news, key=lambda x: x.get('published', ''), reverse=True)

        except Exception as e:
            return [{"error": f"News search failed: {str(e)}"}]

    def research_topic(
        self,
        topic: str,
        include_wiki: bool = True,
        deep_search: bool = False
    ) -> Dict:
        """
        Comprehensive topic research across sources.
        
        :param topic: Topic to research
        :param include_wiki: Whether to include Wikipedia
        :param deep_search: Whether to do deep content search
        :return: Research results from all sources
        """
        try:
            research = {}
            
            # Get Wikipedia information if requested
            if include_wiki:
                wiki_results = self.wiki.search(topic, num_results=1)
                if wiki_results:
                    wiki_page = self.wiki.get_page(wiki_results[0])
                    research["wikipedia"] = {
                        "summary": wiki_page.get("summary", ""),
                        "url": wiki_page.get("url", ""),
                        "references": wiki_page.get("references", [])[:5]
                    }

            # Get academic/research focused results from Exa
            research["academic"] = self.exa.search(
                query=f"{topic} research paper academic",
                num_results=3,
                category="research paper",
                use_autoprompt=True
            ).get('results', [])

            # Get deep content if requested
            if deep_search:
                research["deep_content"] = self.exa.search_with_contents(
                    query=topic,
                    num_results=2,
                    max_chars=2000,
                    highlights=True
                )

            return research

        except Exception as e:
            return {"error": f"Topic research failed: {str(e)}"}

    def fact_check(
        self,
        claim: str,
        thorough: bool = True
    ) -> Dict:
        """
        Fact checking across multiple sources.
        
        :param claim: Claim to fact check
        :param thorough: Whether to do thorough search
        :return: Fact checking results
        """
        try:
            results = {}
            
            # Search fact-checking specific content
            results["fact_check_sources"] = self.exa.search(
                query=f"fact check {claim}",
                num_results=3,
                category="fact check",
                use_autoprompt=True
            ).get('results', [])

            if thorough:
                # Get recent news about the topic
                results["recent_news"] = self.ddg.search_news(
                    query=claim,
                    num_results=3,
                    timelimit="m"
                )

                # Get academic sources
                results["academic_sources"] = self.exa.search(
                    query=f"{claim} research evidence",
                    num_results=2,
                    category="research paper"
                ).get('results', [])

                # Get Wikipedia reference
                wiki_results = self.wiki.search(claim, num_results=1)
                if wiki_results:
                    wiki_summary = self.wiki.get_summary(wiki_results[0], sentences=3)
                    results["wikipedia"] = {
                        "summary": wiki_summary,
                        "title": wiki_results[0]
                    }

            return results

        except Exception as e:
            return {"error": f"Fact checking failed: {str(e)}"}

    def find_similar_content(
        self,
        url: str,
        num_results: int = 3,
        include_content: bool = False
    ) -> List[Dict]:
        """
        Find similar content to a given URL.
        
        :param url: Source URL to find similar content for
        :param num_results: Number of similar results to find
        :param include_content: Whether to include full content
        :return: List of similar content
        """
        try:
            return self.exa.find_similar(
                url=url,
                num_results=num_results,
                with_contents=include_content,
                exclude_source_domain=True
            ).get('results', [])
        except Exception as e:
            return [{"error": f"Similar content search failed: {str(e)}"}]

    # WhatsApp Communication Tools
    def send_whatsapp_message(
        self,
        to: str,
        message: str
    ) -> Dict:
        """
        Send WhatsApp message directly.
        
        :param to: Recipient phone number in E.164 format
        :param message: Message text to send
        :return: Message delivery status
        """
        try:
            return self.twilio.send_message(to, message)
        except Exception as e:
            return {"error": f"Message failed: {str(e)}"}

    def send_whatsapp_image(
        self,
        to: str,
        image_url: str,
        caption: Optional[str] = None
    ) -> Dict:
        """
        Send image via WhatsApp with optional caption.
        
        :param to: Recipient phone number
        :param image_url: URL of image to send
        :param caption: Optional caption text
        :return: Media message status
        """
        try:
            return self.twilio.send_image(to, image_url, caption)
        except Exception as e:
            return {"error": f"Image send failed: {str(e)}"}

    def get_whatsapp_history(
        self,
        max_messages: int = 50
    ) -> List[Dict]:
        """
        Get recent WhatsApp conversation history.
        
        :param max_messages: Maximum messages to retrieve
        :return: List of message objects
        """
        try:
            return self.twilio.get_message_history(limit=max_messages)
        except Exception as e:
            return [{"error": f"History retrieval failed: {str(e)}"}]
